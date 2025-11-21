//
//  TridiagonalAccelerate.swift
//  TridiagonalMatrix
//
//  Created by Joseph Levy on 10/21/25.
//  Updated on 11/21/25 to unify LAPACK signatures.
//

import Accelerate
import Numerics

// Short c pointer aliases
public typealias CMutablePtr<T> = UnsafeMutablePointer<T>
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

// This superset signature includes optional WORK and IWORK arguments.
// - Real implementations will use both.
// - Complex implementations will use WORK and ignore IWORK.
public typealias gtcon<T, M, W> = (
	_ NORM: CMutablePtr<Int8>?, _ N: CMutablePtr<CInt>?,
	_ DL: CMutablePtr<T>?, _ D: CMutablePtr<T>?, _ DU: CMutablePtr<T>?, _ DU2: CMutablePtr<T>?,
	_ IPIV: CMutablePtr<CInt>?, _ anorm: CMutablePtr<M>?, _ rcond: CMutablePtr<M>?,
	_ WORK: CMutablePtr<W>?, _ IWORK: CMutablePtr<CInt>?, // Optional: Backend allocates if nil (ignored by Complex)
	_ info: CMutablePtr<CInt>?
) -> CInt

public typealias axpy<T> = (
	_ n: Int32, _ a: T, _ x: UnsafePointer<T>, _ incx: Int32, _ y: CMutablePtr<T>, _ incy: Int32
) -> Void

// MARK: - Array Extension for Safe Immutable Pointer Access

extension Array {
	@inlinable func withImmutablePointer<R>( _ body: (CMutablePtr<Element>) -> R ) -> R {
		self.withUnsafeBufferPointer { buffer in
			guard let baseAddress = buffer.baseAddress else { preconditionFailure("Array base address is nil") }
			return body(CMutablePtr(mutating: baseAddress))
		}
	}
}

/// Encapsulates generic logic to rebind Swift pointers to LAPACK C-structs
private enum Converters {
	
	@inlinable @discardableResult static func withLUMutables<T, R>(
		dl: inout [T], d: inout [T], du: inout [T], du2: inout [T], ipiv: inout [CInt],
		_ body: (CMutablePtr<T>, CMutablePtr<T>, CMutablePtr<T>, CMutablePtr<T>, CMutablePtr<CInt>) -> R ) -> R {
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
	
	@inline(__always) static func withTridiagonalPointers<T, CType, Result>(
		_ n: Int, _ dl: CMutablePtr<T>, _ d: CMutablePtr<T>, _ du: CMutablePtr<T>, _ du2: CMutablePtr<T>, to type: CType.Type,
		_ body: (CMutablePtr<CType>, CMutablePtr<CType>, CMutablePtr<CType>,  CMutablePtr<CType>) -> Result) -> Result {
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

public protocol ComplexBackend: ScalarField & FloatingPoint {
	associatedtype CType
	associatedtype Real
	static var cgttrf: gttrf<CType> { get }
	static var cgttrs: gttrs<CType> { get }
	static var cgtcon: gtcon<CType, Real, CType> { get }
	static var caxpy: axpy<CType> { get }
}

extension Float: ComplexBackend {
	public typealias CType = __CLPK_complex
	public typealias Real = Float
	public static var cgttrf: gttrf<CType> { cgttrf_ }
	public static var cgttrs: gttrs<CType> { cgttrs_ }
	public static var cgtcon: gtcon<CType, Real, CType> { LAPACKCallBuilder.makeComplexGTCON(cgtcon_) }
	public static var caxpy: axpy<CType> { LAPACKCallBuilder.makeAXPY(cblas_caxpy) }
}

extension Double: ComplexBackend {
	public typealias CType = __CLPK_doublecomplex
	public typealias Real = Double
	public static var cgttrf: gttrf<CType> { zgttrf_ }
	public static var cgttrs: gttrs<CType> { zgttrs_ }
	public static var cgtcon: gtcon<CType, Real,CType> { LAPACKCallBuilder.makeComplexGTCON(zgtcon_) }
	public static var caxpy:  axpy<CType> { LAPACKCallBuilder.makeAXPY(cblas_zaxpy) }
}

// MARK: - Protocol for numeric types used with LAPACK/BLAS

public protocol ScalarField: AlgebraicField where Magnitude: FloatingPoint {
	associatedtype W
	static var one: Self { get }
	static var gttrf: gttrf<Self> { get }
	static var gttrs: gttrs<Self> { get }
	static var gtcon: gtcon<Self, Magnitude, W> { get }
	static var axpy:  axpy<Self>  { get }
}

extension Float: ScalarField {
	public typealias W = Float
	public static var one: Float { 1.0 }
	public static var gttrf: gttrf<Float> { sgttrf_ }
	public static var gttrs: gttrs<Float> { sgttrs_ }
	public static var gtcon: gtcon<Float,Float,Float> { LAPACKCallBuilder.makeRealGTCON(sgtcon_) }
	public static var axpy:  axpy<Float> { cblas_saxpy }
}

extension Double: ScalarField {
	public typealias W = Double
	public static var one: Double { 1.0 }
	public static var gttrf: gttrf<Double> { return dgttrf_ }
	public static var gttrs: gttrs<Double> { return dgttrs_ }
	public static var gtcon: gtcon<Double,Double,Double> { LAPACKCallBuilder.makeRealGTCON(dgtcon_) }
	public static var axpy:  axpy<Double> { cblas_daxpy }
}

// Complex<T> conformances
extension Complex: ScalarField where RealType: ComplexBackend, RealType == RealType.Real {
	public typealias W = RealType.CType
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
	
	public static var gtcon: gtcon<Complex<RealType>,RealType,W>  {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			Converters.withTridiagonalPointers(Int(n!.pointee), dl!, d!, du!, du2!, to: RealType.CType.self) {
				// We pass 'work' to the backend. The backend ignores 'iwork' for Complex, but we pass it for the signature.
				RealType.cgtcon(norm, n, $0 , $1, $2, $3, ipiv, anorm, rcond, work, iwork, info)
			}
		}
	}
	
	public static var axpy: axpy<Complex<RealType>> {
		{ n, a, x, incx, y, incy in
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
	public var lower: [T]
	public var diagonal: [T]
	public var upper: [T]
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
	
	@inlinable public func oneNorm() -> T.Magnitude {
		var norm: T.Magnitude = 0
		let n = size
		if n == 0 { return norm }
		if n == 1 { return diagonal[0].magnitude }
		var col = diagonal[0].magnitude + lower[0].magnitude
		norm = col
		if n > 2 {
			for j in 1..<(n - 1) {
				col = upper[j-1].magnitude + diagonal[j].magnitude + lower[j].magnitude
				if col > norm { norm = col }
			}
		}
		col = upper[n-2].magnitude + diagonal[n-1].magnitude
		if col > norm { norm = col }
		return norm
	}
	
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
	public var lower: [T]
	public var diagonal: [T]
	public var upper: [T]
	public var upper2: [T]
	public var ipiv: [CInt]
	public var rcond: T.Magnitude
	public var anorm: T.Magnitude
	public var isSingular: Bool
	public var count: Int { diagonal.count }
	
	public var approximateConditionNumber: T.Magnitude {
		(!isSingular && rcond > 0) ? 1 / rcond : T.Magnitude.infinity
	}
	
	public var determinant: T = 0
	
	public init(_ A: TridiagonalMatrix<T>, workspace: TridiagonalWorkspace<T>? = nil) {
		self.init(dl: A.lower, d: A.diagonal, du: A.upper, anorm: A.oneNorm(), workspace: workspace)
	}
	
	@inlinable @discardableResult func withLUImmutablePointers<R>(
		_ body: ( _ dl: CMutablePtr<T>, _ d: CMutablePtr<T>, _ du: CMutablePtr<T>, _ du2: CMutablePtr<T>,
				  _ ipiv: CMutablePtr<CInt> ) -> R ) -> R {
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
	
	init(dl: [T], d: [T], du: [T], anorm: T.Magnitude, workspace: TridiagonalWorkspace<T>? = nil) {
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
		var normChar: Int8 = Int8(UnicodeScalar("O").value)
		
		// Helper to handle the Optional Workspace logic cleanly
		func runGTCON(workPtr: CMutablePtr<T.W>?, iworkPtr: CMutablePtr<CInt>?) {
			Converters.withLUMutables(dl: &lower, d: &diagonal, du: &upper, du2: &upper2, ipiv: &ipiv) {
				// Pass the pointers (or nil) to the unified gtcon
				_ = T.gtcon(&normChar, &n_, $0, $1, $2, $3, $4,
							&anorm_, &rcond_,
							workPtr, iworkPtr, // <--- Injected here
							&info)
			}
		}
		// Execution Logic
		if let ws = workspace {
			ws.ensureCapacity(count)
			ws.withPointers { wPtr, iwPtr in
				runGTCON(workPtr: wPtr, iworkPtr: iwPtr)
			}
		} else {
			// Pass nil, letting the Builder allocate internally
			runGTCON(workPtr: nil, iworkPtr: nil)
		}
		
		self.rcond = rcond_
		let detU = diagonal.reduce(T.one, *)
		let signToggle = (0..<(count - 1)).reduce(into: false) { if ipiv[$1] != CInt($1+1) { $0.toggle() } }
		determinant = signToggle ? -detU : detU
	}
	
	// MARK: - Solve
	@discardableResult public func solve(_ b: inout [T]) -> [T] {
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

enum LAPACKCallBuilder {
	
	// Defines the exact signature of the Complex C functions (missing iwork)
	typealias RawComplexGTCON<CType, Real> = (
		_ norm: CMutablePtr<Int8>?, _ n: CMutablePtr<CInt>?,
		_ dl: CMutablePtr<CType>?, _ d: CMutablePtr<CType>?, _ du: CMutablePtr<CType>?, _ du2: CMutablePtr<CType>?,
		_ ipiv: CMutablePtr<CInt>?, _ anorm: CMutablePtr<Real>?, _ rcond: CMutablePtr<Real>?,
		_ work: CMutablePtr<CType>?, _ info: CMutablePtr<CInt>?
	) -> CInt
	
	typealias ComplexAXPY = (
		_ N: Int32, _ a: UnsafeRawPointer?, _ X: UnsafeRawPointer?, _ incX: Int32, _ Y: UnsafeMutableRawPointer?, _ incY: Int32
	) -> Void
	
	/// Returns a 'gtcon' superset closure that wraps the Complex C function.
	/// It accepts 'iwork' (to match the protocol) but ignores it.
	static func makeComplexGTCON<CType, Real>( _ cFunction: @escaping RawComplexGTCON<CType, Real>
	) -> gtcon<CType, Real, CType> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			
			// If user provided workspace, use it
			if let w = work {
				return cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, w, info)
			}
			
			// Allocate temporary workspace
			return withUnsafeTemporaryAllocation(of: CType.self, capacity: max(1, 2 * Int(n!.pointee))) { tempW in
				cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, tempW.baseAddress!, info)
			}
		}
	}
	
	/// Returns a 'gtcon' superset closure that wraps the Real C function.
	/// It passes 'iwork' through to the C function.
	static func makeRealGTCON<CType, Real>( _ cFunction: @escaping gtcon<CType, Real, CType> ) -> gtcon<CType, Real, CType> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			
			// If user provided both workspaces, use them
			if let w = work, let iw = iwork {
				return cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, w, iw, info)
			}
			
			// Allocate temporary workspaces
			return withUnsafeTemporaryAllocation(of: CInt.self, capacity: max(1, Int(n!.pointee))) { tempIW in
				withUnsafeTemporaryAllocation(of: CType.self, capacity: max(1, 2 * Int(n!.pointee))) { tempW in
					let wPtr = work ?? tempW.baseAddress!
					let iwPtr = iwork ?? tempIW.baseAddress!
					return cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, wPtr, iwPtr, info)
				}
			}
		}
	}
	
	static func makeAXPY<CType>( _ cFunction: @escaping ComplexAXPY ) -> axpy<CType> {
		{ n, a, x, incx, y, incy in withUnsafePointer(to: a) { aPtr in cFunction(n, aPtr, x, incx, y, incy) }
		}
	}
}

public final class TridiagonalWorkspace<T: ScalarField> {
	// We can't easily init [T.W] generically without constraints. But we know T.W is a layout-compatible type.
	// Let's use ContiguousArray or just standard Array with a dummy value.
	
	public var work: [T.W]
	public var iwork: [CInt]
	
	public init(capacity: Int = 0) {
		let n = max(1, capacity)
		// We initialize with existing dummy logic or unsafe uninit
		self.work = []
		self.iwork = [CInt](repeating: 0, count: n)
		self.resizeWork(to: 2 * n)
	}
	
	func ensureCapacity(_ n: Int) {
		let reqWork = max(1, 2 * n)
		if work.count < reqWork { resizeWork(to: reqWork) }
		
		let reqIWork = max(1, n)
		if iwork.count < reqIWork { iwork = [CInt](repeating: 0, count: reqIWork) }
	}
	
	private func resizeWork(to count: Int) {
		// Safe way to resize array of unknown T.W
		// We use UnsafeTemporaryAllocation logic essentially but permanent
		work = Array(unsafeUninitializedCapacity: count) { buffer, initializedCount in
			// We strictly don't need to initialize scratch memory,
			// but Swift requires 'init' for safety. Using bzero or memset is fastest.
			bzero(buffer.baseAddress!, count * MemoryLayout<T.W>.stride)
			initializedCount = count
		}
	}
	
	@inlinable func withPointers<R>(_ body: (CMutablePtr<T.W>?, CMutablePtr<CInt>?) -> R) -> R {
		work.withUnsafeMutableBufferPointer { wBuf in
			iwork.withUnsafeMutableBufferPointer { iwBuf in
				body(wBuf.baseAddress, iwBuf.baseAddress)
			}
		}
	}
}
