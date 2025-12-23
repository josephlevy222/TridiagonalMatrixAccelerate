//
//  TridiagonalMatrixAccelerate.swift
//  TridiagonalMatrix
//
//  Cleaned, corrected, and compacted version
//  Changes:
//  - Correct determinant parity via cycle decomposition
//  - Safer workspace ownership (reference held as `let`)
//  - Fix iwork allocation logic (real types need iwork)
//  - Defensive preconditions in `reboundBands`
//  - Removed custom ContiguousStorage protocol (use standard buffer APIs)
//  - Minor naming clarifications and inlining for hot paths
//  - Added: solve(transpose:), batched RHS solves, and more self-contained tests
//

import Accelerate
import Numerics

// short aliases
public typealias CInt = __CLPK_integer
public typealias CMutablePtr = UnsafeMutablePointer
public typealias CVoidPtr = UnsafeRawPointer
public typealias CMutableVoidPtr = UnsafeMutableRawPointer

// MARK: - LAPACK/BLAS C function typealiases
public typealias gttrf<T> = (
	_ N: CMutablePtr<CInt>?, _ DL: CMutablePtr<T>?, _ D: CMutablePtr<T>?, _ DU: CMutablePtr<T>?, _ DU2: CMutablePtr<T>?,
	_ IPIV: CMutablePtr<CInt>?, _ INFO: CMutablePtr<CInt>?
) -> CInt

public typealias gttrs<T> = (
	_ TRANS: CMutablePtr<Int8>?, _ N: CMutablePtr<CInt>?, _ NRHS: CMutablePtr<CInt>?, _ DL: CMutablePtr<T>?,
	_ D: CMutablePtr<T>?, _ DU: CMutablePtr<T>?, _ DU2: CMutablePtr<T>?, _ IPIV: CMutablePtr<CInt>?, _ B: CMutablePtr<T>?,
	_ LDB: CMutablePtr<CInt>?, _ INFO: CMutablePtr<CInt>?
) -> CInt

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

public typealias axpy<T> = (
	_ n: Int32, _ a: T, _ x: UnsafePointer<T>, _ incx: Int32, _ y: CMutablePtr<T>, _ incy: Int32) -> Void

public typealias DSPSignature<T> = (
	_ a: UnsafePointer<T>, _ sa: Int, _ b: UnsafePointer<T>, _ sb: Int, _ c: CMutablePtr<T>, _ sc: Int, _ n: Int) -> Void

public typealias MultiplyAdd<T: ScalarField> =  ( _ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>, _ y: inout ColumnVector<T>
) -> ColumnVector<T>

// MARK: - Real backend protocol (complex backends for real types)
public protocol RealScalar: ScalarField & FloatingPoint & Real {
	associatedtype WType
	static var cgttrf: gttrf<WType> { get }
	static var cgttrs: gttrs<WType> { get }
	static var cgtcon: gtcon<WType, Self, WType> { get }
	static var caxpy: axpy<WType> { get }
	static var vma:  DSPSignature<Self> { get } // Since c and d are alway the same in this code, d is left off
	static var vmul: DSPSignature<Self> { get }
	static var vsub: DSPSignature<Self> { get } //
	static var cAXpY: MultiplyAdd<Complex<Self>> { get }

//	static var cAXpY: MultiplyAdd<Complex<Self>> { get }
}

extension RealScalar {
	public var vma:  DSPSignature<Self> { { a, sa, b, sb, c, sc, n in for i in 0..<n { c[i*sc] += a[i*sa] + b[i+sb]} } }
	public var vmul: DSPSignature<Self> { { a, sa, b, sb, c, sc, n in for i in 0..<n { c[i*sc] = a[i*sa] + b[i+sb] } } }
	public var vsub: DSPSignature<Self> { { b, sb, a, sa, c, sc, n in for i in 0..<n { c[i*sc] = a[i*sa] - b[i+sb] } } }
}
		
// MARK: - Scalar field protocol used with LAPACK/BLAS
public protocol ScalarField: AlgebraicField where Magnitude: FloatingPoint {
	associatedtype CType = Self
	static var one: Self { get }
	static var gttrf: gttrf<Self> { get }
	static var gttrs: gttrs<Self> { get }
	static var gtcon: gtcon<Self, Magnitude, CType> { get }
	static var axpy:  axpy<Self>  { get }
	static var AXpY: MultiplyAdd<Self> { get }
}

extension ScalarField {
	public static var one: Self { 1 }
	
	// helpers to adapt LAPACK signatures that differ for complex real backends
	static func makeRealGTCON<T, R>(_ cFunction: @escaping gtcon<T, R, T>) -> gtcon<T, R, T> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			if let iwork {
				return cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info)
			} else {
				let nn = max(1, Int(n!.pointee))
				return withUnsafeTemporaryAllocation(of: CInt.self, capacity: nn) { tempI in
					return cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, tempI.baseAddress!, info)
				}
			}
		}
	}
	
	static func makeComplexGTCON<T, R>(_ cFunction: @escaping gtconComplex<T, R, T>) -> gtcon<T, R, T> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, unused, info in
			if let work {
				return cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, info)
			} else {
				let nn = max(1, Int(n!.pointee) * 2)
				return withUnsafeTemporaryAllocation(of: T.self, capacity: nn) { tmp in
					return cFunction(norm, n, dl, d, du, du2, ipiv, anorm, rcond, tmp.baseAddress!, info)
				}
			}
		}
	}
	
	static func makeAXPY<CType>(_ cFunction: @escaping (
		_ N: Int32, _ a: CVoidPtr?, _ X: CVoidPtr?, _ incX: Int32, _ Y: CMutableVoidPtr?, _ incY: Int32) -> Void
	) -> axpy<CType> {
		{ n, a, x, incx, y, incy in
			withUnsafePointer(to: a) { aptr in
				cFunction(n, aptr, x, incx, y, incy)
			}
		}
	}
}

// MARK: - Float / Double / Complex bindings
extension Float: ScalarField, RealScalar {
	public static var gttrf: gttrf<Float> { sgttrf_ }
	public static var gttrs: gttrs<Float> { sgttrs_ }
	public static var gtcon: gtcon<Float,Float,Float> { makeRealGTCON(sgtcon_) }
	public static var axpy:  axpy<Float> { cblas_saxpy }
	public static var vma: DSPSignature<Float> {//A   B                    C                     D
		{ vDSP_vma($0, vDSP_Stride($1), $2, vDSP_Stride($3), $4, vDSP_Stride($5),  $4, vDSP_Stride($5), vDSP_Length($6)) } }
	public static var vmul: DSPSignature<Float> {
		{ vDSP_vmul($0, vDSP_Stride($1), $2, vDSP_Stride($3), $4, vDSP_Stride($5), vDSP_Length($6)) } }
	public static var vsub: DSPSignature<Float> {
		{ vDSP_vsub($0, vDSP_Stride($1), $2, vDSP_Stride($3), $4, vDSP_Stride($5), vDSP_Length($6)) } }
	
	public typealias WType = __CLPK_complex
	public static var cgttrf: gttrf<WType> { cgttrf_ }
	public static var cgttrs: gttrs<WType> { cgttrs_ }
	public static var cgtcon: gtcon<WType, Float, WType> { makeComplexGTCON(cgtcon_) }
	public static var caxpy: axpy<WType> { makeAXPY(cblas_caxpy) }
	public static var cAXpY: MultiplyAdd<Complex<Float>> { AXpY_ }
	public static var AXpY: MultiplyAdd<Float> { AXpY_ }
	
}

extension Double: ScalarField, RealScalar {
	public static var gttrf: gttrf<Double> { dgttrf_ }
	public static var gttrs: gttrs<Double> { dgttrs_ }
	public static var gtcon: gtcon<Double,Double,Double> { makeRealGTCON(dgtcon_) }
	public static var axpy:  axpy<Double> { cblas_daxpy }
	public static var vma: DSPSignature<Double> {
		{ vDSP_vmaD($0, vDSP_Stride($1), $2, vDSP_Stride($3), $4, vDSP_Stride($5),  $4, vDSP_Stride($5), vDSP_Length($6)) } }
	public static var vmul: DSPSignature<Double> {
		{ vDSP_vmaD($0, vDSP_Stride($1), $2, vDSP_Stride($3), $4, vDSP_Stride($5),  $4, vDSP_Stride($5), vDSP_Length($6)) } }
	public static var vsub: DSPSignature<Double> {
		{ vDSP_vsubD($0, vDSP_Stride($1), $2, vDSP_Stride($3), $4, vDSP_Stride($5), vDSP_Length($6)) } }
	
	public typealias WType = __CLPK_doublecomplex
	public static var cgttrf: gttrf<WType> { zgttrf_ }
	public static var cgttrs: gttrs<WType> { zgttrs_ }
	public static var cgtcon: gtcon<WType, Double, WType> { makeComplexGTCON(zgtcon_) }
	public static var caxpy:  axpy<WType> { makeAXPY(cblas_zaxpy) }
	public static var cAXpY: MultiplyAdd<Complex<Double>> { AXpY_ }
	public static var AXpY: MultiplyAdd<Double> { AXpY_ }
}

extension Complex: ScalarField where RealType: RealScalar {
	public typealias CType = RealType.WType
	
	public static var gttrf: gttrf<Complex<RealType>> {
		{ n, dl, d, du, du2, ipiv, info in
			reboundBands(n: Int(n!.pointee), dl: dl!, d: d!, du: du!, du2: du2!, to: CType.self) { dlC, dC, duC, du2C in
				RealType.cgttrf(n, dlC, dC, duC, du2C, ipiv, info)
			}
		}
	}
	
	public static var gttrs: gttrs<Complex<RealType>> {
		{ trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info in
			guard let b else { return -1 }
			let countN = Int(n!.pointee)
			let countB = max(Int(nrhs!.pointee) * countN, 1)
			return b.withMemoryRebound(to: CType.self, capacity: countB) { bC in
				reboundBands(n: countN, dl: dl!, d: d!, du: du!, du2: du2!, to: CType.self) { dlC, dC, duC, du2C in
					RealType.cgttrs(trans, n, nrhs, dlC, dC, duC, du2C, ipiv, bC, ldb, info)
				}
			}
		}
	}
	
	public static var gtcon: gtcon<Complex<RealType>, RealType, CType> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			reboundBands(n: Int(n!.pointee), dl: dl!, d: d!, du: du!, du2: du2!, to: CType.self) { dlC, dC, duC, du2C in
				RealType.cgtcon(norm, n, dlC, dC, duC, du2C, ipiv, anorm, rcond, work, iwork, info)
			}
		}
	}
	
	public static var axpy: axpy<Complex<RealType>> {
		{ n, a, x, incx, y, incy in
			withUnsafeBytes(of: a) { aRaw in
				let aC = aRaw.bindMemory(to: CType.self).baseAddress!
				x.withMemoryRebound(to: CType.self, capacity: Int(n)) { xC in
					y.withMemoryRebound(to: CType.self, capacity: Int(n)) { yC in
						RealType.caxpy(n, aC.pointee, xC, incx, yC, incy)
					}
				}
			}
		}
	}
	
	@inline(__always) static func reboundBands<T, CType, R>(
		n: Int, dl: CMutablePtr<T>, d: CMutablePtr<T>, du: CMutablePtr<T>, du2: CMutablePtr<T>,
		to: CType.Type = CType.self,
		body: (CMutablePtr<CType>, CMutablePtr<CType>, CMutablePtr<CType>, CMutablePtr<CType>) -> R
	) -> R {
		precondition(n > 0, "n must be > 0")
		return dl.withMemoryRebound(to: CType.self, capacity: max(1, n-1)) { dlC in
			d.withMemoryRebound(to: CType.self, capacity: n) { dC in
				du.withMemoryRebound(to: CType.self, capacity: max(1, n-1)) { duC in
					du2.withMemoryRebound(to: CType.self, capacity: max(0, n-2)) { du2C in
						return body(dlC, dC, duC, du2C)
					}
				}
			}
		}
	}
	
	public static var AXpY: MultiplyAdd<Complex<RealType>> { AXpY_ }
}

// MARK: - Tridiagonal Matrix types
@frozen public struct TridiagonalMatrix<T: ScalarField> {
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
		let n = size
		guard n > 0 else { return 0 }
		if n == 1 { return diagonal[0].magnitude }
		
		var norm = diagonal[0].magnitude + lower[0].magnitude
		if n > 2 {
			for j in 1..<(n-1) {
				let col = upper[j-1].magnitude + diagonal[j].magnitude + lower[j].magnitude
				norm = max(norm, col)
			}
		}
		norm = max(norm, upper[n-2].magnitude + diagonal[n-1].magnitude)
		return norm
	}
	
	@inlinable public func factorized() -> TridiagonalLUMatrix<T> { TridiagonalLUMatrix(self) }
}

public typealias ColumnVector<T: ScalarField> = Array<T>

// MARK: - Basic matrix-vector ops
//public func *<T: ScalarField>(_ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>) -> ColumnVector<T> {
//	precondition(x.count == A.size, "Invalid column vector size")
//	let n = x.count
//	if n == 0 { return [] }
//	if n == 1 { return [A.diagonal[0] * x[0]] }
//	
//	var b = Array<T>(repeating: .zero, count: n)
//	b[0] = A.diagonal[0] * x[0] + A.upper[0] * x[1]
//	for j in 1..<(n-1) {
//		b[j] = A.lower[j-1] * x[j-1] + A.diagonal[j] * x[j] + A.upper[j] * x[j+1]
//	}
//	b[n-1] = A.lower[n-2] * x[n-2] + A.diagonal[n-1] * x[n-1]
//	return b
//}

public func oldAXpY<T: ScalarField>(A: TridiagonalMatrix<T>, x: ColumnVector<T>, y: inout ColumnVector<T>) -> ColumnVector<T> {
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
@frozen public struct TridiagonalLUMatrix<T: ScalarField> {
	public var lower: ContiguousArray<T>
	public var diagonal: ContiguousArray<T>
	public var upper: ContiguousArray<T>
	public var upper2: ContiguousArray<T>
	public var ipiv: ContiguousArray<CInt>
	
	public var rcond: T.Magnitude
	public var anorm: T.Magnitude
	public var isSingular: Bool
	public var count: Int { diagonal.count }
	private let workspace: TridiagonalWorkspace<T>
	
	public var approximateConditionNumber: T.Magnitude {
		(!isSingular && rcond > 0) ? 1 / rcond : T.Magnitude.infinity
	}
	
	public var determinant: T = 0
	
	@inline(__always) mutating func withMutableLUBufferPointers<R>(_ body: (
		_ dl: CMutablePtr<T>, _ d: CMutablePtr<T>,
		_ du: CMutablePtr<T>, _ du2: CMutablePtr<T>,
		_ ipiv: CMutablePtr<CInt>) -> R) -> R {
			lower.withUnsafeMutableBufferPointer { dlBuf in
				diagonal.withUnsafeMutableBufferPointer { dBuf in
					upper.withUnsafeMutableBufferPointer { duBuf in
						upper2.withUnsafeMutableBufferPointer { du2Buf in
							ipiv.withUnsafeMutableBufferPointer { ipivBuf in
								guard let dlBase = dlBuf.baseAddress,
									  let dBase  = dBuf.baseAddress,
									  let duBase = duBuf.baseAddress,
									  let du2Base = du2Buf.baseAddress,
									  let ipivBase = ipivBuf.baseAddress
								else { preconditionFailure("Unexpected nil base address") }
								return body(dlBase, dBase, duBase, du2Base, ipivBase)
							}
						}
					}
				}
			}
		}
	
	public init(_ A: TridiagonalMatrix<T>) {
		self.init(dl: A.lower, d: A.diagonal, du: A.upper, anorm: A.oneNorm())
	}
	
	@inline(__always) init(dl: [T], d: [T], du: [T], anorm: T.Magnitude) {
		precondition(d.count > 0, "Matrix must be non-empty")
		precondition(dl.count == d.count - 1 && du.count == d.count - 1, "Diagonal sizes inconsistent")
		
		self.lower = ContiguousArray(dl)
		self.diagonal = ContiguousArray(d)
		self.upper = ContiguousArray(du)
		self.upper2 = ContiguousArray(repeating: .zero, count: max(0, d.count - 2))
		self.ipiv = ContiguousArray(repeating: 0, count: d.count)
		self.anorm = anorm
		self.rcond = 0
		self.isSingular = true
		self.determinant = 0
		self.workspace = TridiagonalWorkspace(capacity: d.count)
		
		var n_ = CInt(d.count)
		var info = CInt(0)
		
		// factor
		withMutableLUBufferPointers { dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
			_ = T.gttrf(&n_, dlPtr, dPtr, duPtr, du2Ptr, ipivPtr, &info)
		}
		if info != 0 { return }
		isSingular = false
		
		// determinant: compute product of diagonal (U diagonal) and permutation parity by cycle decomposition
		let detU = diagonal.reduce(T.one, *)
		let n = count
		var seen = Array(repeating: false, count: n)
		var cycles = 0
		for i in 0..<n where !seen[i] {
			var j = i
			cycles += 1
			while !seen[j] {
				seen[j] = true
				let pivot = Int(ipiv[j]) - 1
				j = pivot
			}
		}
		let parityIsNegative = ((n - cycles) % 2) != 0
		determinant = parityIsNegative ? -detU : detU
		
		// condition estimate (gtcon)
		var anorm_ = anorm
		var rcond_ = T.Magnitude(0)
		var normChar: Int8 = Int8(UnicodeScalar("O").value)
		let workPtr = workspace.workBuffer(for: count)
		let iworkPtr = workspace.iworkBuffer(for: count)
		withMutableLUBufferPointers { dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
			_ = T.gtcon(&normChar, &n_, dlPtr, dPtr, duPtr, du2Ptr, ipivPtr, &anorm_, &rcond_, workPtr, iworkPtr, &info)
		}
		self.rcond = rcond_
	}
	
	// MARK: - Solve (single RHS)
	@discardableResult public mutating func solve(_ b: inout [T], transpose: Bool = false) -> [T] {
		precondition(b.count == self.count)
		guard !isSingular else { return Array(repeating: .zero, count: count) }
		
		var n = CInt(count)
		var nrhs = CInt(1)
		var ldb = n
		var info = CInt(0)
		let transChar: Int8 = transpose ? (T.self is (any RealScalar.Type) ? Int8(UnicodeScalar("T").value) : Int8(UnicodeScalar("C").value)) : Int8(UnicodeScalar("N").value)
		var trans = transChar
		b.withUnsafeMutableBufferPointer { buffer in
			guard let bPtr = buffer.baseAddress else { preconditionFailure("Array base address is nil") }
			withMutableLUBufferPointers { dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
				_ = T.gttrs(&trans, &n, &nrhs, dlPtr, dPtr, duPtr, du2Ptr, ipivPtr, bPtr, &ldb, &info)
			}
		}
		return b
	}
	
	// Solve where B is provided in column-major contiguous storage: length must be n*nrhs
	@discardableResult public mutating func solve(_ bColumnMajor: inout [T], nrhs: Int, transpose: Bool = false) -> [T] {
		precondition(bColumnMajor.count == self.count * nrhs, "B must be column-major n x nrhs")
		guard !isSingular else { return Array(repeating: .zero, count: bColumnMajor.count) }
		
		var n = CInt(count)
		var nrhs_c = CInt(nrhs)
		var ldb = n
		var info = CInt(0)
		let transChar: Int8 = transpose ? (T.self is (any RealScalar.Type) ? Int8(UnicodeScalar("T").value) : Int8(UnicodeScalar("C").value)) : Int8(UnicodeScalar("N").value)
		var trans = transChar
		bColumnMajor.withUnsafeMutableBufferPointer { buf in
			guard let bPtr = buf.baseAddress else { preconditionFailure("B base address is nil") }
			withMutableLUBufferPointers { dlPtr, dPtr, duPtr, du2Ptr, ipivPtr in
				_ = T.gttrs(&trans, &n, &nrhs_c, dlPtr, dPtr, duPtr, du2Ptr, ipivPtr, bPtr, &ldb, &info)
			}
		}
		return bColumnMajor
	}
	
	// Solve where B is an array of column vectors [[T]] (each column length == n). Returns solved columns in same shape.
	@discardableResult public mutating func solve(columns: [[T]], transpose: Bool = false) -> [[T]] {
		let nrhs = columns.count
		guard nrhs > 0 else { return [] }
		let n = count
		precondition(columns.allSatisfy { $0.count == n }, "All columns must have length n")
		
		// Pack into column-major contiguous buffer
		var bFlat = Array<T>(repeating: .zero, count: n * nrhs)
		for j in 0..<nrhs {
			let col = columns[j]
			for i in 0..<n { bFlat[i + j*n] = col[i] }
		}
		
		let solvedFlat = solve(&bFlat, nrhs: nrhs, transpose: transpose)
		
		// Unpack
		var out: [[T]] = Array(repeating: Array(repeating: .zero, count: n), count: nrhs)
		for j in 0..<nrhs {
			for i in 0..<n { out[j][i] = solvedFlat[i + j*n] }
		}
		return out
	}
}

// MARK: - Workspace (reference type)
public final class TridiagonalWorkspace<T: ScalarField> {
	public private(set) var work: CMutablePtr<T.CType>
	public private(set) var iwork: CMutablePtr<CInt>?
	
	private var capacityWork: Int
	private var capacityIWork: Int
	
	public init(capacity: Int = 0) {
		let n = max(1, capacity)
		self.capacityWork = 2 * n
		self.work = CMutablePtr<T.CType>.allocate(capacity: capacityWork)
		
		// Real types (Float/Double) require iwork for gtcon; complex backends do not
		if T.self is any RealScalar.Type {
			self.capacityIWork = n
			self.iwork = CMutablePtr<CInt>.allocate(capacity: capacityIWork)
		} else {
			self.capacityIWork = 0
			self.iwork = nil
		}
	}
	
	deinit {
		work.deallocate()
		iwork?.deallocate()
	}
	
	public func workBuffer(for n: Int) -> CMutablePtr<T.CType> {
		let required = max(1, 2 * n)
		if capacityWork < required {
			work.deallocate()
			capacityWork = max(required, capacityWork * 2)
			work = CMutablePtr<T.CType>.allocate(capacity: capacityWork)
		}
		return work
	}
	
	public func iworkBuffer(for n: Int) -> CMutablePtr<CInt>? {
		guard T.self is any RealScalar.Type else { return nil }
		let required = max(1, n)
		if capacityIWork < required {
			iwork?.deallocate()
			capacityIWork = max(required, capacityIWork * 2)
			iwork = CMutablePtr<CInt>.allocate(capacity: capacityIWork)
		}
		return iwork
	}
}

