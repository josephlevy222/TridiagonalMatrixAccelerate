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

import Accelerate
import Numerics

// MARK: - LAPACK/BLAS C Function Declarations & Types

public typealias CLPKInteger = __CLPK_integer

public typealias gttrf<T> = (
	_ N: UnsafeMutablePointer<CLPKInteger>?,
	_ DL: UnsafeMutablePointer<T>?, _ D: UnsafeMutablePointer<T>?, _ DU: UnsafeMutablePointer<T>?,
	_ DU2: UnsafeMutablePointer<T>?, _ IPIV: UnsafeMutablePointer<CLPKInteger>?,
	_ INFO: UnsafeMutablePointer<CLPKInteger>?
) -> CLPKInteger

public typealias gttrs<T> = (
	_ TRANS: UnsafeMutablePointer<Int8>?, _ N: UnsafeMutablePointer<CLPKInteger>?, _ NRHS: UnsafeMutablePointer<CLPKInteger>?,
	_ DL: UnsafeMutablePointer<T>?, _ D: UnsafeMutablePointer<T>?, _ DU: UnsafeMutablePointer<T>?,
	_ DU2: UnsafeMutablePointer<T>?, _ IPIV: UnsafeMutablePointer<CLPKInteger>?, _ B: UnsafeMutablePointer<T>?,
	_ LDB: UnsafeMutablePointer<CLPKInteger>?, _ INFO: UnsafeMutablePointer<CLPKInteger>?
) -> CLPKInteger

public typealias gtcon<T> = (
	_ NORM: UnsafeMutablePointer<Int8>?, _ N: UnsafeMutablePointer<CLPKInteger>?, _ DL: UnsafeMutablePointer<T>?,
	_ D: UnsafeMutablePointer<T>?, _ DU: UnsafeMutablePointer<T>?, _ DU2: UnsafeMutablePointer<T>?,
	_ IPIV: UnsafeMutablePointer<CLPKInteger>?, _ ANORM: UnsafeMutableRawPointer?, _ RCOND: UnsafeMutableRawPointer?,
	_ INFO: UnsafeMutablePointer<CLPKInteger>?
) -> CLPKInteger // Note work and iwork left for implementation

public typealias axpy<T> = (
	_ n: Int32, _ a: T, _ x: UnsafePointer<T>, _ incx: Int32, _ y: UnsafeMutablePointer<T>, _ incy: Int32
) -> Void

// MARK: - Protocol for numeric types used with LAPACK/BLAS

public protocol ScalarField: AlgebraicField where Magnitude: FloatingPoint {
	static var one: Self { get }
	static var gttrf: gttrf<Self> { get }
	static var gttrs: gttrs<Self> { get }
	static var gtcon: gtcon<Self> { get }
	static var axpy:  axpy<Self>  { get }
}

// MARK: - Array Extension for Safe Immutable Pointer Access

extension Array {
	/// Provides a temporary UnsafeMutablePointer from an Array instance. This is safe for C-API calls (like LAPACK gttrs) that require a mutable pointer
	@inlinable func withImmutablePointer<R>( _ body: (UnsafeMutablePointer<Element>) -> R ) -> R {
		self.withUnsafeBufferPointer { buffer in
			///The use of UnsafeMutablePointer(mutating:) ensures safety by providing  a temporary pointer that does not modify the conceptual immutability of the array
			guard let baseAddress = buffer.baseAddress else { preconditionFailure("Array base address is nil") }
			return body(UnsafeMutablePointer(mutating: baseAddress))
		}
	}
}

/// Encapsulates generic logic to rebind Swift pointers to LAPACK C-structs
/// avoiding deep nesting in the main protocol conformances.
private enum LapackMemory {
	
	// MARK: - TridiagonalLUMatrix Pointer Access Helpers (Integrated)
	
	/// Internal free function helper to consolidate the nested `withUnsafeMutableBufferPointer` calls for the five LU arrays.
	/// Safely used in `TridiagonalLUMatrix.init` via `inout` parameters.
	@inlinable @discardableResult static func withLUMutables<T, R>(
		dl: inout [T], d: inout [T], du: inout [T], du2: inout [T], ipiv: inout [CLPKInteger],
		_ body: (UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>,
				 UnsafeMutablePointer<T>, UnsafeMutablePointer<CLPKInteger>) -> R ) -> R {
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
		_ n: Int, _ dl: UnsafeMutablePointer<T>, _ d: UnsafeMutablePointer<T>, _ du: UnsafeMutablePointer<T>,
		_ du2: UnsafeMutablePointer<T>, to type: CType.Type,
		_ body: (UnsafeMutablePointer<CType>, UnsafeMutablePointer<CType>, UnsafeMutablePointer<CType>,
				 UnsafeMutablePointer<CType>) -> Result) -> Result {
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
	
	/// Rebinds tridiagonal arrays AND the RHS vector `b` to `CType`.
	@inline(__always) static func withTridiagonalAndRHS<T, CType, Result>(
		_ n: Int, _ nrhs: Int, _ dl: UnsafeMutablePointer<T>, _ d: UnsafeMutablePointer<T>, _ du: UnsafeMutablePointer<T>,
		_ du2: UnsafeMutablePointer<T>, _ b: UnsafeMutablePointer<T>, to type: CType.Type,
		_ body: (UnsafeMutablePointer<CType>, UnsafeMutablePointer<CType>, UnsafeMutablePointer<CType>,
				 UnsafeMutablePointer<CType>, UnsafeMutablePointer<CType>) -> Result
	) -> Result {
		withTridiagonalPointers(n, dl, d, du, du2, to: type) { dlC, dC, duC, du2C in
			b.withMemoryRebound(to: type, capacity: max(1,  n * max(1, nrhs))) { bC in
				body(dlC, dC, duC, du2C, bC)
			}
		}
	}
}

// MARK: - Conformances for Float / Double / Complex

extension Float: ScalarField {
	public static var one: Float { 1.0 }
	public static var gttrf: gttrf<Float> { return sgttrf_ }
	public static var gttrs: gttrs<Float> { return sgttrs_ }
	public static var gtcon: gtcon<Float> {
		{ norm, n, dl, d, du, du2, ipiv, anormRaw, rcondRaw, info in
			let N = Int(n!.pointee)
			var work = [Float](repeating: 0.0, count: max(1, 2 * N))
			var iwork = [CLPKInteger](repeating: 0, count: max(1, N))
			return work.withUnsafeMutableBufferPointer { workPtr in
				iwork.withUnsafeMutableBufferPointer { iworkPtr in
					sgtcon_(norm, n, dl, d, du, du2, ipiv, anormRaw?.assumingMemoryBound(to: Float.self),
							rcondRaw?.assumingMemoryBound(to: Float.self), workPtr.baseAddress, iworkPtr.baseAddress, info)
				}
			}
		}
	}
	public static var axpy: axpy<Float> { { n, a, x, incx, y, incy in cblas_saxpy(n, a, x, incx, y, incy) } }
}

extension Double: ScalarField {
	public static var one: Double { 1.0 }
	public static var gttrf: gttrf<Double> { return dgttrf_ }
	public static var gttrs: gttrs<Double> { return dgttrs_ }
	public static var gtcon: gtcon<Double> {
		{ norm, n, dl, d, du, du2, ipiv, anormRaw, rcondRaw, info in
			let N = Int(n!.pointee)
			// Revert to local allocation for safety/simplicity with Float/Double
			var work = [Double](repeating: 0.0, count: max(1, 2 * N))
			var iwork = [CLPKInteger](repeating: 0, count: max(1, N))
			return work.withUnsafeMutableBufferPointer { workPtr in
				iwork.withUnsafeMutableBufferPointer { iworkPtr in
					dgtcon_(norm, n, dl, d, du, du2, ipiv, anormRaw?.assumingMemoryBound(to: Double.self),
							rcondRaw?.assumingMemoryBound(to: Double.self), workPtr.baseAddress, iworkPtr.baseAddress, info)
				}
			}
		}
	}
	public static var axpy: axpy<Double> { { n, a, x, incx, y, incy in cblas_daxpy(n, a, x, incx, y, incy) } }
}

// Complex<T> conformances
extension Complex: ScalarField where RealType: FloatingPoint {
	public static var one: Complex<RealType> { Complex(1, 0) }
	public static var gttrf: gttrf<Complex<RealType>> {
		{ n, dl, d, du, du2, ipiv, info in
			let countN = Int(n!.pointee)
			return if RealType.self == Float.self {
				LapackMemory.withTridiagonalPointers(countN, dl!, d!, du!, du2!, to: __CLPK_complex.self) {
					cgttrf_(n, $0, $1, $2, $3, ipiv, info)
				}
			} else if RealType.self == Double.self {
				LapackMemory.withTridiagonalPointers(countN, dl!, d!, du!, du2!, to: __CLPK_doublecomplex.self) {
					zgttrf_(n, $0, $1, $2, $3, ipiv, info)
				}
			} else {
				fatalError("gttrf not supported for Complex<\(RealType.self)>")
			}
		}
	}
	
	public static var gttrs: gttrs<Complex<RealType>> {
		{ trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info in
			let countN = Int(n!.pointee)
			let countRHS = Int(nrhs!.pointee)
			
			return if RealType.self == Float.self {
				LapackMemory.withTridiagonalAndRHS(countN, countRHS, dl!, d!, du!, du2!, b!, to: __CLPK_complex.self) {
					cgttrs_(trans, n, nrhs, $0, $1, $2, $3, ipiv, $4, ldb, info)
				}
			} else if RealType.self == Double.self {
				LapackMemory.withTridiagonalAndRHS(countN, countRHS, dl!, d!, du!, du2!, b!, to: __CLPK_doublecomplex.self) {
					zgttrs_(trans, n, nrhs, $0, $1, $2, $3, ipiv, $4, ldb, info)
				}
			} else {
				fatalError("gttrs not supported for Complex<\(RealType.self)>")
			}
		}
	}
	
	public static var gtcon: gtcon<Complex<RealType>> {
		{ norm, n, dl, d, du, du2, ipiv, anormRaw, rcondRaw, info in
			let N = Int(n!.pointee)
			
			return if RealType.self == Float.self {
				// 1. Rebind matrix pointers
				LapackMemory.withTridiagonalPointers(N, dl!, d!, du!, du2!, to: __CLPK_complex.self) { dlC, dC, duC, du2C in
					var work = [__CLPK_complex](repeating: __CLPK_complex(), count: max(1, 2 * N))
					return work.withUnsafeMutableBufferPointer { w in
						cgtcon_(norm, n, dlC, dC, duC, du2C, ipiv,
								anormRaw?.assumingMemoryBound(to: Float.self),
								rcondRaw?.assumingMemoryBound(to: Float.self),
								w.baseAddress!, info)
					}
				}
			} else if RealType.self == Double.self {
				LapackMemory.withTridiagonalPointers(N, dl!, d!, du!, du2!, to: __CLPK_doublecomplex.self) {dlC,dC,duC,du2C in
					var work = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(), count: max(1, 2 * N))
					return work.withUnsafeMutableBufferPointer { w in
						zgtcon_(norm, n, dlC, dC, duC, du2C, ipiv,
								anormRaw?.assumingMemoryBound(to: Double.self),
								rcondRaw?.assumingMemoryBound(to: Double.self),
								w.baseAddress!, info)
					}
				}
			} else {
				fatalError("gtcon not supported for Complex<\(RealType.self)>")
			}
		}
	}
	
	public static var axpy: axpy<Complex<RealType>> {
		// AXPY for complex still requires manual rebinding for a, x, y.
		if RealType.self == Float.self {
			return { n, a, x, incx, y, incy in
				withUnsafePointer(to: a) { aPtr in
					aPtr.withMemoryRebound(to: __CLPK_complex.self, capacity: 1) { aC in
						x.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n)) { xC in
							y.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n)) { yC in
								cblas_caxpy(n, aC, xC, incx, yC, incy)
							}
						}
					}
				}
			}
		} else if RealType.self == Double.self {
			return { n, a, x, incx, y, incy in
				withUnsafePointer(to: a) { aPtr in
					aPtr.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: 1) { aC in
						x.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n)) { xC in
							y.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n)) { yC in
								cblas_zaxpy(n, aC, xC, incx, yC, incy)
							}
						}
					}
				}
			}
		} else {
			fatalError("axpy not supported")
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
	public var ipiv: [CLPKInteger] // pivot indices
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
		_ body: ( _ dl: UnsafeMutablePointer<T>, _ d: UnsafeMutablePointer<T>, _ du: UnsafeMutablePointer<T>,
				  _ du2: UnsafeMutablePointer<T>, _ ipiv: UnsafeMutablePointer<CLPKInteger> ) -> R ) -> R {
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
		ipiv = [CLPKInteger](repeating: 0, count: d.count)
		self.anorm = anorm
		rcond = 0
		isSingular = true
		determinant = 0
		
		var n_ = CLPKInteger(count)
		var info = CLPKInteger(0)
		
		// 1. FACTOR (gttrf)
		LapackMemory.withLUMutables(dl: &lower, d: &diagonal, du: &upper, du2: &upper2, ipiv: &ipiv) {
			_ = T.gttrf(&n_, $0, $1, $2, $3, $4, &info)
		}
		
		if info != 0 { return }
		
		// 2. CONDITION ESTIMATE (gtcon)
		var anorm_ = anorm
		var rcond_ = T.Magnitude(0)
		var normChar: Int8 = Int8(UnicodeScalar("O").value) // 'O' one-norm
		
		withUnsafeMutablePointer(to: &anorm_) { anormPtr in
			withUnsafeMutablePointer(to: &rcond_) { rcondPtr in
				LapackMemory.withLUMutables(dl: &lower, d: &diagonal, du: &upper, du2: &upper2, ipiv: &ipiv) {
					_ = T.gtcon(&normChar, &n_, $0, $1, $2, $3, $4, // dlPtr, dPtr, duPtr, du2Ptr, ipivPtr,
								UnsafeMutableRawPointer(anormPtr), UnsafeMutableRawPointer(rcondPtr), &info)
				}
			}
		}
		
		if info != 0 { return }
		
		self.rcond = rcond_
		// determinant: product of diagonal of U with sign determined by pivot permutations
		let detU = diagonal.reduce(T.one, *)
		var signToggle = false
		for i in 0..<(count - 1) where ipiv[i] != CLPKInteger(i + 1) { signToggle.toggle() }
		determinant = signToggle ? -detU : detU
	}
	
	// MARK: - Solve
	/// Solve A * x = b using precomputed LU. Returns solved vector or throws on failure.
	@discardableResult public func solve(_ b: inout [T]) -> [T] { // LAPACK expects in-out array
		precondition(b.count == self.count)
		guard !isSingular else { return Array(repeating: .zero, count: count)}
		
		var n_ = CLPKInteger(count)
		var nrhs = CLPKInteger(1)
		var ldb = n_
		var info = CLPKInteger(0)
		var trans: Int8 = Int8(UnicodeScalar("N").value)
		
		withLUImmutablePointers { dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
			b.withImmutablePointer { bPtr in
				_ = T.gttrs(&trans, &n_, &nrhs, dlPtr, dPtr, duPtr, du2Ptr, ipivPtr, bPtr, &ldb, &info)
			}
		}
		return b
	}
}
