//
//  TridiagonalAccelerate.swift
//  TridiagonalMatrix
//
//  Created by Joseph Levy on 10/21/25.
//  Updated on 11/21/2025 to unify LAPACK signatures.
//
//
import Accelerate
import Numerics

// MARK: - Short c pointer aliases
public typealias CMutablePtr<T> = UnsafeMutablePointer<T>
public typealias CInt = __CLPK_integer
public typealias CMutableVoidPtr = UnsafeMutableRawPointer
public typealias CVoidPtr = UnsafeRawPointer
 
// MARK: - Contiguous storage protocol
public protocol ContiguousStorage {
	associatedtype Element
	@inlinable mutating func withContiguousMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R
	@inlinable func withContiguousBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R
}

extension Array: ContiguousStorage {
	@inlinable mutating public func withContiguousMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R {
		return try self.withUnsafeMutableBufferPointer { try body($0) }
	}
	@inlinable public func withContiguousBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
		return try self.withUnsafeBufferPointer { try body($0) }
	}
}

extension ContiguousArray: ContiguousStorage {
	@inlinable mutating public func withContiguousMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R {
		return try self.withUnsafeMutableBufferPointer { try body($0) }
	}
	@inlinable public func withContiguousBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
		return try self.withUnsafeBufferPointer { try body($0) }
	}
}

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

// This signature includes optional WORK and IWORK arguments.
// - Real implementations will use both.
// - Complex implementations will use WORK and ignore IWORK. With gtconComplex matching the LAPACK signature.
public typealias gtcon<T, M, W> = (
	_ NORM: CMutablePtr<Int8>?, _ N: CMutablePtr<CInt>?, _ DL: CMutablePtr<T>?, _ D: CMutablePtr<T>?,
	_ DU: CMutablePtr<T>?, _ DU2: CMutablePtr<T>?, _ IPIV: CMutablePtr<CInt>?, _ anorm: CMutablePtr<M>?,
	_ rcond: CMutablePtr<M>?, _ WORK: CMutablePtr<W>?, _ IWORK: CMutablePtr<CInt>?, _ info: CMutablePtr<CInt>?
) -> CInt

public typealias gtconComplex<T, M, W> = (
	_ NORM: CMutablePtr<Int8>?, _ N: CMutablePtr<CInt>?, _ DL: CMutablePtr<T>?, _ D: CMutablePtr<T>?,
	_ DU: CMutablePtr<T>?, _ DU2: CMutablePtr<T>?, _ IPIV: CMutablePtr<CInt>?, _ anorm: CMutablePtr<M>?,
	_ rcond: CMutablePtr<M>?, _ WORK: CMutablePtr<W>?, _ info: CMutablePtr<CInt>?
) -> CInt

public typealias axpy<T> = (_ n: Int32, _ a: T, _ x: UnsafePointer<T>, _ incx: Int32, _ y: CMutablePtr<T>, _ incy: Int32)
-> Void

// MARK: - Backend protocol for complex precision bindings
public protocol ComplexScalar: ScalarField & FloatingPoint { // Type that can be used in Complex<T>
	associatedtype WType
	static var cgttrf: gttrf<WType> { get }
	static var cgttrs: gttrs<WType> { get }
	static var cgtcon: gtcon<WType, Self, WType> { get }
	static var caxpy: axpy<WType> { get }
}

// MARK: - Protocol for numeric types used with LAPACK/BLAS
public protocol ScalarField: AlgebraicField where Magnitude: FloatingPoint {
	associatedtype CType = Self
	static var one: Self { get }
	static var gttrf: gttrf<Self> { get }
	static var gttrs: gttrs<Self> { get }
	static var gtcon: gtcon<Self, Magnitude, CType> { get }
	static var axpy:  axpy<Self>  { get }
}

extension ScalarField {
	public static var one: Self { 1 }
}

extension Float: ScalarField, ComplexScalar {
	public static var gttrf: gttrf<Float> { sgttrf_ }
	public static var gttrs: gttrs<Float> { sgttrs_ }
	public static var gtcon: gtcon<Float,Float,Float> { AccelerateCallBuilder.makeRealGTCON(sgtcon_) }
	public static var axpy:  axpy<Float> { cblas_saxpy }
	
	public typealias WType = __CLPK_complex
	public static var cgttrf: gttrf<WType> { cgttrf_ }
	public static var cgttrs: gttrs<WType> { cgttrs_ }
	public static var cgtcon: gtcon<WType, Float, WType> { AccelerateCallBuilder.makeComplexGTCON(cgtcon_) }
	public static var caxpy: axpy<WType> { AccelerateCallBuilder.makeAXPY(cblas_caxpy) }
}

extension Double: ScalarField, ComplexScalar {
	public static var gttrf: gttrf<Double> { dgttrf_ }
	public static var gttrs: gttrs<Double> { dgttrs_ }
	public static var gtcon: gtcon<Double,Double,Double> { AccelerateCallBuilder.makeRealGTCON(dgtcon_) }
	public static var axpy:  axpy<Double> { cblas_daxpy }
	
	public typealias WType = __CLPK_doublecomplex
	public static var cgttrf: gttrf<WType> { zgttrf_ }
	public static var cgttrs: gttrs<WType> { zgttrs_ }
	public static var cgtcon: gtcon<WType, Double, WType> { AccelerateCallBuilder.makeComplexGTCON(zgtcon_) }
	public static var caxpy:  axpy<WType> { AccelerateCallBuilder.makeAXPY(cblas_zaxpy) }
}

extension Complex: ScalarField where RealType: ComplexScalar { // LAPACK calls must convert Complex to CLPKType
	public typealias CType = RealType.WType
	public static var gttrf: gttrf<Complex<RealType>> {
		{ n, dl, d, du, du2, ipiv, info in
			reboundBands(n: Int(n!.pointee), dl: dl!, d: d!, du: du!, du2: du2!, to: CType.self) {
				RealType.cgttrf(n, $0, $1, $2, $3, ipiv, info)
			}
		}
	}

	public static var gttrs: gttrs<Complex<RealType>> {
		{ trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info in
			guard let b else {return -1}
			let countN = Int(n!.pointee)
			let countB = max(Int(nrhs!.pointee)*countN, 1)
			return b.withMemoryRebound(to: CType.self, capacity: countB) { bC in
				reboundBands(n: Int(n!.pointee), dl: dl!, d: d!, du: du!, du2: du2!, to: CType.self) {
					RealType.cgttrs(trans, n, nrhs, $0, $1, $2, $3, ipiv, bC, ldb, info)
				}
			}
		}
	}

	public static var gtcon: gtcon<Complex<RealType>,RealType,CType>  {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			reboundBands(n: Int(n!.pointee), dl: dl!, d: d!, du: du!, du2: du2!, to: CType.self) {
				// Pass 'work' to the backend. The backend ignores 'iwork' for Complex, but we pass it for the signature.
				RealType.cgtcon(norm, n, $0, $1, $2, $3, ipiv, anorm, rcond, work, iwork, info)
			}
		}
	}

	public static var axpy: axpy<Complex<RealType>> {
		{ n, a, x, incx, y, incy in
			withUnsafePointer(to: a) { aPtr in
				aPtr.withMemoryRebound(to:CType.self, capacity: 1) { aC in
					x.withMemoryRebound(to: CType.self, capacity: Int(n)) { xC in
						y.withMemoryRebound(to: CType.self, capacity: Int(n)) { yC in
							RealType.caxpy(n, aC.pointee, xC, incx, yC, incy)
						}
					}
				}
			}
		}
	}
	
	/// Encapsulates generic logic to rebind Swift pointers to LAPACK C-structs
	@inline(__always) static func reboundBands<T, CType, R>(
		n: Int,
		dl: CMutablePtr<T>, d: CMutablePtr<T>, du: CMutablePtr<T>, du2: CMutablePtr<T>,
		to: CType.Type = CType.self,
		body: (CMutablePtr<CType>, CMutablePtr<CType>, CMutablePtr<CType>, CMutablePtr<CType>) -> R
	) -> R {
		dl.withMemoryRebound(to: CType.self, capacity: max(1, n-1)) { dl in
			d.withMemoryRebound(to: CType.self, capacity: n) { d in
				du.withMemoryRebound(to: CType.self, capacity: max(1, n-1)) { du in
					du2.withMemoryRebound(to: CType.self, capacity: max(0, n-2)) { du2 in
						body(dl, d, du, du2)
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
	public var lower: ContiguousArray<T>
	public var diagonal: ContiguousArray<T>
	public var upper: ContiguousArray<T>
	public var upper2: ContiguousArray<T>
	public var ipiv: ContiguousArray<CInt>
	
	public var rcond: T.Magnitude
	public var anorm: T.Magnitude
	public var isSingular: Bool
	public var count: Int { diagonal.count }
	private var workspace = TridiagonalWorkspace<T>()

	public var approximateConditionNumber: T.Magnitude {
		(!isSingular && rcond > 0) ? 1 / rcond : T.Magnitude.infinity
	}

	public var determinant: T = 0

	public init(_ A: TridiagonalMatrix<T>) {
		self.init(dl: A.lower, d: A.diagonal, du: A.upper, anorm: A.oneNorm())
	}
	
	@inlinable mutating func withMutableLUBufferPointers<R>(_ body: (
		_ dl: UnsafeMutablePointer<T>, _ d: UnsafeMutablePointer<T>,
		_ du: UnsafeMutablePointer<T>, _ du2: UnsafeMutablePointer<T>,
		_ ipiv: UnsafeMutablePointer<CInt>) -> R) -> R {
		lower.withContiguousMutableBufferPointer { dlBuf in
			diagonal.withContiguousMutableBufferPointer { dBuf in
				upper.withContiguousMutableBufferPointer { duBuf in
					upper2.withContiguousMutableBufferPointer { du2Buf in
						ipiv.withContiguousMutableBufferPointer { ipivBuf in
							guard let dlBase = dlBuf.baseAddress,
								  let dBase  = dBuf.baseAddress,
								  let duBase = duBuf.baseAddress,
								  let du2Base = du2Buf.baseAddress,
								  let ipivBase = ipivBuf.baseAddress else {
								preconditionFailure("Unexpected nil base address")
							}
							return body(dlBase, dBase, duBase, du2Base, ipivBase)
						}
					}
				}
			}
		}
	}

	init(dl: [T], d: [T], du: [T], anorm: T.Magnitude) { //}, workspace: TridiagonalWorkspace<T>? = nil) {
		precondition(d.count > 0, "Matrix must be non-empty")
		precondition(dl.count == d.count - 1 && du.count == d.count - 1, "Diagonal sizes inconsistent")

		self.lower = ContiguousArray(dl)
		self.diagonal = ContiguousArray(d)
		self.upper = ContiguousArray(du)
		self.upper2 = ContiguousArray(repeating: .zero, count: max(0, d.count - 2))
		self.ipiv = ContiguousArray(repeating: 0, count: d.count)
		self.anorm = anorm
		rcond = 0
		isSingular = true
		determinant = 0

		var n_ = CInt(count)
		var info = CInt(0)

		// 1. FACTOR (gttrf)
		withMutableLUBufferPointers { //dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
			_ = T.gttrf(&n_, $0, $1, $2, $3, $4, &info)
		}

		if info != 0 { return }

		// 2. Calculate Determinant
		let detU = diagonal.reduce(T.one, *)
		let signToggle = (0..<(count - 1)).reduce(into: false) { if ipiv[$1] != CInt($1+1) { $0.toggle() } }
		determinant = signToggle ? -detU : detU

		// 3. CONDITION ESTIMATE (gtcon)
		var anorm_ = anorm
		var rcond_ = T.Magnitude(0)
		var normChar: Int8 = Int8(UnicodeScalar("O").value)
		let workPtr = workspace.workBuffer(for: count)     // T.CType*
		let iworkPtr = workspace.iworkBuffer(for: count)   // CInt* or nil
		withMutableLUBufferPointers { //dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
			_ = T.gtcon(&normChar, &n_, $0, $1, $2, $3, $4, &anorm_, &rcond_, workPtr, iworkPtr, &info)
		}
		self.rcond = rcond_
	}

	// MARK: - Solve
	@discardableResult public mutating func solve(_ b: inout [T]) -> [T] { // not really mutatings
		precondition(b.count == self.count)
		guard !isSingular else { return Array(repeating: .zero, count: count)}

		var n = CInt(count)
		var nrhs = CInt(1)
		var ldb = n
		var info = CInt(0)
		var trans: Int8 = Int8(UnicodeScalar("N").value)
		b.withContiguousMutableBufferPointer { buffer in
			guard let bPtr =  buffer.baseAddress else { preconditionFailure("Array base address is nil") 	}
			withMutableLUBufferPointers { //dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
				_ = T.gttrs(&trans, &n, &nrhs, $0, $1, $2, $3, $4, bPtr, &ldb, &info)
			}
		}
		return b
	}
}

enum AccelerateCallBuilder {
	/// Returns a 'gtcon' superset closure that wraps the Complex C function.
	/// It accepts 'iwork' (to match the protocol) but ignores it.
	static func makeRealGTCON<T, R>( _ cFunction: @escaping gtcon<T, R, T> ) -> gtcon<T, R, T> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			// Define the IWORK resolution logic (Allocates only if needed)
			func withResolvedIWork(_ work: CMutablePtr<T>) -> CInt {
				if let iwork { cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info) }
				else {
					withUnsafeTemporaryAllocation(of: CInt.self, capacity: max(1, Int(n!.pointee))) {
						cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, $0.baseAddress!, info)
					}
				}
			}
			// Main Body: Resolve WORK and enter the chain
			return if let work {
				 withResolvedIWork(work)
			} else {
				withUnsafeTemporaryAllocation(of: T.self, capacity: max(1, 2 * Int(n!.pointee))) {
					withResolvedIWork($0.baseAddress!)
				}
			}
		}
	}

	static func makeComplexGTCON<T, R>( _ cFunction: @escaping gtconComplex<T, R, T> ) -> gtcon<T, R, T> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, unused, info in
			 if let work {
				cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, info)
			} else {
				withUnsafeTemporaryAllocation(of: T.self, capacity: max(1, 2 * Int(n!.pointee))) {
					cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, $0.baseAddress!, info)
				}
			}
		}
	}

	// Type to match cblas_caxpy
	static func makeAXPY<CType>(_ cFunction: @escaping (
		_ N: Int32, _ a: CVoidPtr?, _ X: CVoidPtr?, _ incX: Int32, _ Y: CMutableVoidPtr?, _ incY: Int32) -> Void
	) -> axpy<CType> {/**/{ n, a, x, incx, y, incy in withUnsafePointer(to: a) { cFunction(n, $0, x, incx, y, incy) } }/**/}
}

public final class TridiagonalWorkspace<T: ScalarField> {

	public private(set) var work: CMutablePtr<T.CType>
	public private(set) var iwork: CMutablePtr<CInt>?

	private var capacityWork: Int
	private var capacityIWork: Int

	public init(capacity: Int = 0) {
		let n = max(1, capacity)
		self.capacityWork = 2 * n
		self.work = CMutablePtr<T.CType>.allocate(capacity: capacityWork)

		// Only allocate iwork for real fields
		if T.self is (any ComplexScalar) {
			self.capacityIWork = 0
			self.iwork = nil
		} else {
			self.capacityIWork = n
			self.iwork = CMutablePtr<CInt>.allocate(capacity: capacityIWork)
		}
	}

	deinit {
		work.deallocate()
		iwork?.deallocate()
	}

	/// RAII-style accessor: always returns a valid work buffer
	public func workBuffer(for n: Int) -> CMutablePtr<T.CType> {
		let required = max(1, 2 * n)
		if capacityWork < required {
			work.deallocate()
			capacityWork = max(required, capacityWork * 2)
			work = CMutablePtr<T.CType>.allocate(capacity: capacityWork)
		}
		return work
	}

	/// RAII-style accessor: returns iwork only for real fields
	public func iworkBuffer(for n: Int) -> CMutablePtr<CInt>? {
		guard !(T.self is(any ComplexScalar)) else { return nil }
		let required = max(1, n)
		if capacityIWork < required {
			iwork?.deallocate()
			capacityIWork = max(required, capacityIWork * 2)
			iwork = CMutablePtr<CInt>.allocate(capacity: capacityIWork)
		}
		return iwork
	}
}

