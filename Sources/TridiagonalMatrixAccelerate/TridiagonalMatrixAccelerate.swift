//
//  TridiagonalAccelerate.swift
//  TridiagonalMatrix
//
//  Created by Joseph Levy on 10/21/25.
//
//  Extended on 10/22/25 to include general matrix definitions
//  and arithmetic operations from TridiagonalMatrix.swift.
//
//  Upgraded on 10/22/25 to use Accelerate/BLAS for aXpY.
//
//  Refactored on 10/25/25 to unify protocols, add condition number,
//  and integrate factorization into the initializer.
//
//  - Fixed problem that Complex<Double> conflicted with Complex<Float> by checking RealType at runtime
//    and returning the correct LAPACK routine
//
//  Using Gemini and ChatGPT tightened the code between 10/25/25 and 11/20/25
//
//  TridiagonalMatrixAccelerate.swift — updated to use a backend protocol for Complex<T>
//  Rewrites Complex conformances to avoid runtime RealType checks
//  Added func builders for Complex compliance

import Accelerate
import Numerics

// Short c pointer aliases — used only in Swift-facing APIs and helpers
public typealias CMutablePtr<T> = UnsafeMutablePointer<T>
public typealias RawPtr = UnsafeMutableRawPointer
public typealias CInt = __CLPK_integer


// MARK: - LAPACK/BLAS C Function Declarations & Types

public typealias gttrf<T> = (
	_ N: CMutablePtr<CInt>?, _ DL: CMutablePtr<T>?, _ D: CMutablePtr<T>?, _ DU: CMutablePtr<T>?, _ DU2: CMutablePtr<T>?,
	_ IPIV: CMutablePtr<CInt>?, _ INFO: CMutablePtr<CInt>?
) -> CInt

public typealias gttrs<T> = (
	_ TRANS: CMutablePtr<Int8>?, _ N: CMutablePtr<CInt>?, _ NRHS: CMutablePtr<CInt>?, _ DL: CMutablePtr<T>?,
	_ D: CMutablePtr<T>?, _ DU: CMutablePtr<T>?, _ DU2: CMutablePtr<T>?, _ IPIV: CMutablePtr<CInt>?, _ B: CMutablePtr<T>?,
	_ LDB: CMutablePtr<CInt>?, _ INFO: CMutablePtr<CInt>?
) -> CInt

public typealias gtcon<T, M> = (
	_ NORM: CMutablePtr<Int8>?, _ N: CMutablePtr<CInt>?, _ DL: CMutablePtr<T>?, _ D: CMutablePtr<T>?,
	_ DU: CMutablePtr<T>?, _ DU2: CMutablePtr<T>?, _ IPIV: CMutablePtr<CInt>?, _ anorm: CMutablePtr<M>?,
	_ rcond: CMutablePtr<M>?, _ info: CMutablePtr<CInt>?) -> CInt // Note work and iwork left for implementation

public typealias axpy<T> = (
	_ n: Int32, _ a: T, _ x: UnsafePointer<T>, _ incx: Int32, _ y: CMutablePtr<T>, _ incy: Int32
) -> Void

// MARK: - Array Extension for Safe Immutable Pointer Access

extension Array {
	/// Provides a temporary MutablePtr from an Array instance. This is safe for C-API calls (like LAPACK gttrs) that require a mutable pointer
	@inlinable func withImmutablePointer<R>( _ body: (CMutablePtr<Element>) -> R ) -> R {
		self.withUnsafeBufferPointer { buffer in
			///The use of CMutablePtr(mutating:) ensures safety by providing  a temporary pointer that does not modify the conceptual immutability of the array
			guard let baseAddress = buffer.baseAddress else { preconditionFailure("Array base address is nil") }
			return body(CMutablePtr(mutating: baseAddress))
		}
	}
}

/// Encapsulates generic logic to rebind Swift pointers to LAPACK C-structs
/// avoiding deep nesting in the main protocol conformances.
private enum Converters { // a namespace
	
	// MARK: - TridiagonalLUMatrix Pointer Access Helpers (Integrated)
	
	/// Internal free function helper to consolidate the nested `withUnsafeMutableBufferPointer` calls for the five LU arrays.
	/// Safely used in `TridiagonalLUMatrix.init` via `inout` parameters.
	@inlinable @discardableResult static func withLUMutables<T, R>(
		dl: inout [T], d: inout [T], du: inout [T], du2: inout [T], ipiv: inout [CInt],
		_ body: (CMutablePtr<T>, CMutablePtr<T>, CMutablePtr<T>, CMutablePtr<T>, CMutablePtr<CInt>) -> R ) -> R {
		// Access is safe because arrays are passed 'inout'
		dl.withUnsafeMutableBufferPointer { dlBP in
			d.withUnsafeMutableBufferPointer { dBP in
				du.withUnsafeMutableBufferPointer { duBP in
					du2.withUnsafeMutableBufferPointer { du2BP in
						ipiv.withUnsafeMutableBufferPointer { ipivBP in
							body(dlBP.baseAddress!,dBP.baseAddress!,duBP.baseAddress!,du2BP.baseAddress!,ipivBP.baseAddress!)
						}
					}
				}
			}
		}
	}
	
	/// Rebinds the 4 tridiagonal arrays from type `T` to `CType` and executes the body.
	/// Used for converting Complex<Float> -> __CLPK_complex, etc.
	@inline(__always) static func withTridiagonalPointers<T, CType, Result>(
		_ n: Int, _ dl: CMutablePtr<T>, _ d: CMutablePtr<T>, _ du: CMutablePtr<T>, _ du2: CMutablePtr<T>, to type: CType.Type,
		_ body: (CMutablePtr<CType>, CMutablePtr<CType>, CMutablePtr<CType>,  CMutablePtr<CType>) -> Result) -> Result {
			// We use max(1, ...) to ensure we don't bind capacity 0, which can be unsafe in some contexts
			dl.withMemoryRebound(to: type, capacity: max(1, n - 1)) { dlC in
				d.withMemoryRebound(to: type, capacity: max(1, n)) { dC in
					du.withMemoryRebound(to: type, capacity: max(1, n - 1)) { duC in
						du2.withMemoryRebound(to: type, capacity: max(1, n - 2)) { du2C in
							body(dlC, dC, duC, du2C)
						}
					}
				}
			}
		}
}

// MARK: - Backend protocol for complex precision bindings

public protocol ComplexBackend: ScalarField & FloatingPoint { // Had to make public
	associatedtype CType
	associatedtype Real
	static var cgttrf: gttrf<CType> { get }
	static var cgttrs: gttrs<CType> { get }
	static var cgtcon: gtcon<CType, Real> { get }
	static var caxpy: axpy<CType> { get }
}

// Implementations for Float/Double complex backends
extension Float: ComplexBackend {
	public typealias CType = __CLPK_complex
	public typealias Real = Float
	public static var cgttrf: gttrf<CType> { cgttrf_ }
	public static var cgttrs: gttrs<CType> { cgttrs_ }
	public static var cgtcon: gtcon<CType, Real> { LAPACKCallBuilder.makeComplexGTCON(cgtcon_) }
	public static var caxpy: axpy<CType> { LAPACKCallBuilder.makeAXPY(cblas_caxpy) }
}

extension Double: ComplexBackend {
	public typealias CType = __CLPK_doublecomplex
	public typealias Real = Double
	public static var cgttrf: gttrf<CType> { zgttrf_ }
	public static var cgttrs: gttrs<CType> { zgttrs_ }
	public static var cgtcon: gtcon<CType, Real> { LAPACKCallBuilder.makeComplexGTCON(zgtcon_) }
	public static var caxpy:  axpy<CType> { LAPACKCallBuilder.makeAXPY(cblas_zaxpy) }
}

// MARK: - Protocol for numeric types used with LAPACK/BLAS

public protocol ScalarField: AlgebraicField where Magnitude: FloatingPoint {
	static var one: Self { get }
	static var gttrf: gttrf<Self> { get }
	static var gttrs: gttrs<Self> { get }
	static var gtcon: gtcon<Self, Magnitude> { get }
	static var axpy:  axpy<Self>  { get }
}

// MARK: - Conformances for Float / Double / Complex

extension Float: ScalarField {
	public static var one: Float { 1.0 }
	public static var gttrf: gttrf<Float> { sgttrf_ }
	public static var gttrs: gttrs<Float> { sgttrs_ }
	public static var gtcon: gtcon<Float,Float> { LAPACKCallBuilder.makeRealGTCON(sgtcon_) }
	public static var axpy:  axpy<Float> { cblas_saxpy }
}

extension Double: ScalarField {
	public static var one: Double { 1.0 }
	public static var gttrf: gttrf<Double> { return dgttrf_ }
	public static var gttrs: gttrs<Double> { return dgttrs_ }
	public static var gtcon: gtcon<Double,Double> { LAPACKCallBuilder.makeRealGTCON(dgtcon_) }
	public static var axpy:  axpy<Double> { cblas_daxpy }
}

// Complex<T> conformances — now use backend protocol to avoid runtime testing
extension Complex: ScalarField where RealType: ComplexBackend, RealType == RealType.Real {
	
	public static var one: Complex<RealType> { Complex(1, 0) }
	
	public static var gttrf: gttrf<Complex<RealType>> {
		{ n, dl, d, du, du2, ipiv, info in
			Converters.withTridiagonalPointers(Int(n!.pointee), dl!, d!, du!, du2!, to: RealType.CType.self) {
				RealType.cgttrf(n, $0, $1, $2, $3, ipiv, info)
			}
		}
	}
	
	public static var gttrs: gttrs<Complex<RealType>> {
		{ trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info in
			guard let b else {return -1}
			let countN = Int(n!.pointee)
			let countRHS = Int(nrhs!.pointee)
			return b.withMemoryRebound(to: RealType.CType.self, capacity: max(1, countN * max(1, countRHS))) { bC in
				return Converters.withTridiagonalPointers(countN, dl!, d!, du!, du2!, to: RealType.CType.self) {
					RealType.cgttrs(trans, n, nrhs, $0, $1, $2, $3, ipiv, bC, ldb, info)
				}
			}
		}
	}
	
	public static var gtcon: gtcon<Complex<RealType>,RealType>  {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, info in
			Converters.withTridiagonalPointers(Int(n!.pointee), dl!, d!, du!, du2!, to: RealType.CType.self) {
				RealType.cgtcon(norm, n, $0 , $1, $2, $3, ipiv, anorm, rcond, info)
			}
		}
	}
	
	public static var axpy: axpy<Complex<RealType>> {
		{ n, a, x, incx, y, incy in
			// Rebind a, x, y to the backend CType and call RealType.axpy
			withUnsafePointer(to: a) { aPtr in
				aPtr.withMemoryRebound(to: RealType.CType.self, capacity: 1) { aC in
					x.withMemoryRebound(to: RealType.CType.self, capacity: Int(n)) { xC in
						y.withMemoryRebound(to: RealType.CType.self, capacity: Int(n)) { yC in
							RealType.caxpy(n, aC.pointee, xC, incx, yC, incy)
						}
					}
				}
			}
		}
	}
}

// MARK: - Tridiagonal Matrix & Vector Types

public struct TridiagonalMatrix<T: ScalarField> {
	public var lower: [T]     // size n-1
	public var diagonal: [T]  // size n
	public var upper: [T]     // size n-1
	public let size: Int
	
	public init(diagonal: [T], upper: [T], lower: [T]) {
		precondition(diagonal.count > 0, "Diagonal must not be empty")
		precondition(diagonal.count == upper.count + 1, "Invalid upper size")
		precondition(diagonal.count == lower.count + 1, "Invalid lower size")
		self.diagonal = diagonal
		self.upper = upper
		self.lower = lower
		self.size = diagonal.count
	}
	
	/// 1-norm (maximum column sum). Uses `magnitude` on scalars.
	@inlinable public func oneNorm() -> T.Magnitude {
		var norm: T.Magnitude = 0
		let n = size
		if n == 0 { return norm }
		if n == 1 { return diagonal[0].magnitude }
		// first column
		var col = diagonal[0].magnitude + lower[0].magnitude
		norm = col
		// middle
		if n > 2 {
			for j in 1..<(n - 1) {
				col = upper[j-1].magnitude + diagonal[j].magnitude + lower[j].magnitude
				if col > norm { norm = col }
			}
		}
		// last
		col = upper[n-2].magnitude + diagonal[n-1].magnitude
		if col > norm { norm = col }
		return norm
	}
	
	/// Convenience: create a factorization object
	public func factorized() -> TridiagonalLUMatrix<T> {
		TridiagonalLUMatrix(self)
	}
}

public typealias ColumnVector<T: ScalarField> = Array<T>

// MARK: - Basic matrix-vector ops

public func *<T: ScalarField>(_ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>) -> ColumnVector<T> {
	precondition(x.count == A.size, "Invalid column vector size")
	let n = x.count
	if n == 0 { return [] }
	if n == 1 { return [A.diagonal[0] * x[0]] }
	
	var b = Array<T>(repeating: .zero, count: n)
	b[0] = A.diagonal[0] * x[0] + A.upper[0] * x[1]
	for j in 1..<(n-1) {
		b[j] = A.lower[j-1] * x[j-1] + A.diagonal[j] * x[j] + A.upper[j] * x[j+1]
	}
	b[n-1] = A.lower[n-2] * x[n-2] + A.diagonal[n-1] * x[n-1]
	return b
}

public func AXpY<T: ScalarField>(A: TridiagonalMatrix<T>, x: ColumnVector<T>, y: ColumnVector<T>) -> ColumnVector<T> {
	precondition(x.count == A.size && y.count == A.size, "Vector sizes must match")
	let n = x.count
	if n == 0 { return [] }
	if n == 1 { return [y[0] + A.diagonal[0] * x[0]] }
	
	var b = y
	b[0] += A.diagonal[0] * x[0] + A.upper[0] * x[1]
	for j in 1..<(n-1) {
		b[j] += A.lower[j-1] * x[j-1] + A.diagonal[j] * x[j] + A.upper[j] * x[j+1]
	}
	b[n-1] += A.lower[n-2] * x[n-2] + A.diagonal[n-1] * x[n-1]
	return b
}

// AXPY wrappers using BLAS
public func aXpY_Inplace<T: ScalarField>(a: T, x: ColumnVector<T>, y: inout ColumnVector<T>) {
	precondition(x.count == y.count, "Vector size mismatch")
	let n = Int32(x.count)
	if n == 0 { return }
	T.axpy(n, a, x, 1, &y, 1)
}

public func aXpY<T: ScalarField>(a: T, x: ColumnVector<T>, y: ColumnVector<T>) -> ColumnVector<T> {
	var out = y
	aXpY_Inplace(a: a, x: x, y: &out)
	return out
}

// MARK: - LU Factorization Container

public struct TridiagonalLUMatrix<T: ScalarField> {
	public var lower: [T]   // DL (n-1) - stored LU sub-diagonal (modified by gttrf)
	public var diagonal: [T]  // D  (n)   - stored LU main diagonal (modified by gttrf)
	public var upper: [T] // DU (n-1) - stored LU super-diagonal (modified by gttrf)
	public var upper2: [T] // DU2 (n-2)
	public var ipiv: [CInt] // pivot indices
	public var rcond: T.Magnitude  // estimate reciprocal condition number
	public var anorm: T.Magnitude  // 1-norm of original matrix
	public var isSingular: Bool    // Flags singular
	public var count: Int { diagonal.count }
	
	public var approximateConditionNumber: T.Magnitude {
		(!isSingular && rcond > 0) ? 1 / rcond : T.Magnitude.infinity
	}
	
	public var determinant: T = 0
	
	/// Attempt to factorize
	public init(_ A: TridiagonalMatrix<T>) {
		self.init(dl: A.lower, d: A.diagonal, du: A.upper, anorm: A.oneNorm())
	}
	
	@inlinable @discardableResult func withLUImmutablePointers<R>(
		_ body: ( _ dl: CMutablePtr<T>, _ d: CMutablePtr<T>, _ du: CMutablePtr<T>, _ du2: CMutablePtr<T>,
				  _ ipiv: CMutablePtr<CInt> ) -> R ) -> R {
		// All five arrays now use the single, generic helper function
		lower.withImmutablePointer { dlPtr in
			diagonal.withImmutablePointer { dPtr in
				upper.withImmutablePointer { duPtr in
					upper2.withImmutablePointer { du2Ptr in
						ipiv.withImmutablePointer { ipivPtr in
							body(dlPtr, dPtr, duPtr, du2Ptr, ipivPtr)
						}
					}
				}
			}
		}
	}
	
	init(dl: [T], d: [T], du: [T], anorm: T.Magnitude) {
		precondition(d.count > 0, "Matrix must be non-empty")
		precondition(dl.count == d.count - 1 && du.count == d.count - 1, "Diagonal sizes inconsistent")
		
		lower = dl
		diagonal = d
		upper = du
		upper2 = [T](repeating: .zero, count: max(0, d.count - 2))
		ipiv = [CInt](repeating: 0, count: d.count)
		self.anorm = anorm
		rcond = 0
		isSingular = true
		determinant = 0
		
		var n_ = CInt(count)
		var info = CInt(0)
		
		// 1. FACTOR (gttrf)
		Converters.withLUMutables(dl: &lower, d: &diagonal, du: &upper, du2: &upper2, ipiv: &ipiv) {
			_ = T.gttrf(&n_, $0, $1, $2, $3, $4, &info)
		}
		
		if info != 0 { return }
		
		// 2. CONDITION ESTIMATE (gtcon)
		var anorm_ = anorm
		var rcond_ = T.Magnitude(0)
		var normChar: Int8 = Int8(UnicodeScalar("O").value) // 'O' one-norm
		
		withUnsafeMutablePointer(to: &anorm_) { anormPtr in
			withUnsafeMutablePointer(to: &rcond_) { rcondPtr in
				Converters.withLUMutables(dl: &lower, d: &diagonal, du: &upper, du2: &upper2, ipiv: &ipiv) {
					_ = T.gtcon(&normChar, &n_, $0, $1, $2, $3, $4, anormPtr, rcondPtr, &info) // dl, dPtr, du, du2, ipiv,
				}
			}
		}
		
		if info != 0 { return }
		
		self.rcond = rcond_
		// determinant: product of diagonal of U with sign determined by pivot permutations
		let detU = diagonal.reduce(T.one, *)
		let signToggle = (0..<(count - 1)).reduce(into: false) { if ipiv[$1] != CInt($1+1) { $0.toggle() } }
		determinant = signToggle ? -detU : detU
	}
	
	// MARK: - Solve
	/// Solve A * x = b using precomputed LU. Returns solved vector or throws on failure.
	@discardableResult public func solve(_ b: inout [T]) -> [T] { // LAPACK expects in-out array
		precondition(b.count == self.count)
		guard !isSingular else { return Array(repeating: .zero, count: count)}
		
		var n = CInt(count)
		var nrhs = CInt(1)
		var ldb = n
		var info = CInt(0)
		var trans: Int8 = Int8(UnicodeScalar("N").value)
		
		withLUImmutablePointers { dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
			b.withImmutablePointer { bPtr in
				_ = T.gttrs(&trans, &n, &nrhs, dlPtr, dPtr, duPtr, du2Ptr, ipivPtr, bPtr, &ldb, &info)
			}
		}
		return b
	}
}

// Matches sgtcon_ and dgtcon_
public typealias RealGTCON<CType, Real> = (
	_ norm: CMutablePtr<Int8>?, _ n: CMutablePtr<CInt>?,
	_ dl: CMutablePtr<CType>?, _ d: CMutablePtr<CType>?, _ du: CMutablePtr<CType>?, _ du2: CMutablePtr<CType>?,
	_ ipiv: CMutablePtr<CInt>?, _ anorm: CMutablePtr<Real>?, _ rcond: CMutablePtr<Real>?,
	_ work: CMutablePtr<CType>?, _ iwork: CMutablePtr<CInt>?, _ info: CMutablePtr<CInt>?
) -> CInt

// Matches cgtcon_ and zgtcon_
public typealias ComplexGTCON<CType, Real> = (
	_ norm: CMutablePtr<Int8>?, _ n: CMutablePtr<CInt>?,
	_ dl: CMutablePtr<CType>?, _ d: CMutablePtr<CType>?, _ du: CMutablePtr<CType>?, _ du2: CMutablePtr<CType>?,
	_ ipiv: CMutablePtr<CInt>?, _ anorm: CMutablePtr<Real>?, _ rcond: CMutablePtr<Real>?,
	_ work: CMutablePtr<CType>?, _ info: CMutablePtr<CInt>?
) -> CInt

// Matches cblas_caxpy and cblas_zaxpy
// Note: Accelerate defines 'a', 'X', 'Y' as UnsafeRawPointer/UnsafeMutableRawPointer
public typealias ComplexAXPY = (
	_ N: Int32, _ a: UnsafeRawPointer?, _ X: UnsafeRawPointer?, _ incX: Int32, _ Y: UnsafeMutableRawPointer?, _ incY: Int32
) -> Void

enum LAPACKCallBuilder {
	
	/// Returns a closure that handles workspace allocation (or injection) automatically
	static func makeComplexGTCON<CType, Real>( _ cFunction: @escaping ComplexGTCON<CType, Real> ) -> gtcon<CType, Real> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, info in // work  removed for now
			withUnsafeTemporaryAllocation(of: CType.self, capacity: max(1, 2 * Int(n!.pointee))) { tempW in
				cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, tempW.baseAddress!, info)
			}
		}
	}
	
	static func makeRealGTCON<CType, Real>( _ cFunction: @escaping RealGTCON<CType, Real> ) -> gtcon<CType, Real> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, info in // work and iwork removed for now
			withUnsafeTemporaryAllocation(of: CInt.self, capacity: max(1, Int(n!.pointee))) { iworkPtr in
				withUnsafeTemporaryAllocation(of: CType.self, capacity: max(1, 2 * Int(n!.pointee))) { tempW in
					cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, tempW.baseAddress!, iworkPtr.baseAddress!, info)
				}
			}
		}
	}
	
	/// Returns a closure that handles the unsafe pointer casting for BLAS AXPY
	static func makeAXPY<CType>( _ cFunction: @escaping ComplexAXPY ) -> axpy<CType> {
		// Convert typed pointers (CType) a to RawPointer for CBLAS
		{ n, a, x, incx, y, incy in withUnsafePointer(to: a) { aPtr in cFunction(n, aPtr, x, incx, y, incy) }
		}
	}
}
