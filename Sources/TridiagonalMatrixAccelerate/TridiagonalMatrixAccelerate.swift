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
import Accelerate
//import simd
import Numerics

// MARK: - LAPACK C Function Declarations GTTRF(Factor), GTTRS(Solve), GTCON(Condition Number)
public typealias CLPKInteger = __CLPK_integer

public typealias gttrf<T> = (
	_ N: UnsafeMutablePointer<CLPKInteger>?,
	_ DL: UnsafeMutablePointer<T>?,            _ D: UnsafeMutablePointer<T>?,             _  DU: UnsafeMutablePointer<T>?,
	_ DU2: UnsafeMutablePointer<T>?,           _ IPIV: UnsafeMutablePointer<CLPKInteger>?,
    _ INFO: UnsafeMutablePointer<CLPKInteger>?) -> CLPKInteger

public typealias gttrs<T> = (
	_ TRANS: UnsafeMutablePointer<Int8>?, _ N: UnsafeMutablePointer<CLPKInteger>?, _ NRHS: UnsafeMutablePointer<CLPKInteger>?,
	_ DL: UnsafeMutablePointer<T>?,            _ D: UnsafeMutablePointer<T>?,             _  DU: UnsafeMutablePointer<T>?,
	_ DU2: UnsafeMutablePointer<T>?,           _ IPIV: UnsafeMutablePointer<CLPKInteger>?,_ B: UnsafeMutablePointer<T>?,
	_ LDB: UnsafeMutablePointer<CLPKInteger>?, _ INFO: UnsafeMutablePointer<CLPKInteger>?
) -> CLPKInteger

public typealias gtcon<T> = (
	_ NORM: UnsafeMutablePointer<Int8>?,      _  N: UnsafeMutablePointer<CLPKInteger>?, _ DL: UnsafeMutablePointer<T>?,
	_ D: UnsafeMutablePointer<T>?,             _ DU: UnsafeMutablePointer<T>?,          _ DU2: UnsafeMutablePointer<T>?,
	_ IPIV: UnsafeMutablePointer<CLPKInteger>?, _ ANORM: UnsafeMutableRawPointer?,      _ RCOND: UnsafeMutableRawPointer?,
	_ WORK: UnsafeMutablePointer<T>?,          _ iwork: UnsafeMutablePointer<CLPKInteger>?, _ INFO: UnsafeMutablePointer<CLPKInteger>?
) -> CLPKInteger

public typealias axpy<T> = (
	_ n: Int32, _ a: T, _ x: UnsafePointer<T>, _ incx: Int32, _ y: UnsafeMutablePointer<T>, _ incy: Int32
) -> Void

/// A unified protocol for types that can be used in both general arithmetic
/// and BLAS/LAPACK-based solving.
public protocol ScalarField: AlgebraicField where Magnitude: FloatingPoint {
	static var one: Self { get }
	static var gttrf: gttrf<Self> { get }
	static var gttrs: gttrs<Self> { get }
	static var gtcon: gtcon<Self> { get }
	static var axpy: axpy<Self> { get }
	//static func axpy(n: Int32, a: Self, x: UnsafePointer<Self>, incx: Int32, y: UnsafeMutablePointer<Self>, incy: Int32)
}

// MARK: - Conformances for Real and Complex Types

extension Float: ScalarField {
	public static var one: Float { Float(1.0) }
	public static var gttrf: gttrf<Float> { return sgttrf_ }
	public static var gttrs: gttrs<Float> { return sgttrs_ }
	public static var gtcon: gtcon<Float> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			sgtcon_(norm, n, dl, d, du, du2, ipiv, anorm?.assumingMemoryBound(to: Float.self),
					rcond?.assumingMemoryBound(to: Float.self), work, iwork, info)
		}
	}
	public static var axpy: axpy<Float> {
		{ n, a, x, incx, y, incy in cblas_saxpy(n, a, x, incx, y, incy) }
	}
}

extension Double: ScalarField {
	public static var one: Double { 1.0 }
	public static var gttrf: gttrf<Double> { return dgttrf_ }
	public static var gttrs: gttrs<Double> { return dgttrs_ }
	public static var gtcon: gtcon<Double> {
		{ norm, n, dl, d, du, du2, ipiv, anorm, rcond, work, iwork, info in
			dgtcon_(norm, n, dl, d, du, du2, ipiv, anorm?.assumingMemoryBound(to: Double.self),
					rcond?.assumingMemoryBound(to: Double.self), work, iwork, info)
		}
	}
	public static var axpy: axpy<Double> {
		{ n, a, x, incx, y, incy in cblas_daxpy(n, a, x, incx, y, incy) }
	}
}

extension Complex: ScalarField  {
	public static var one: Complex<RealType> { Complex(1, 0) }
	
	public static var gttrf: gttrf<Complex<RealType>> {
		(RealType.self == Float.self) ?
		{ n, dl, d, du, du2, ipiv, info in
			let result = dl!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee) - 1) { dl in
				d!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee)) { d in
					du!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee) - 1) { du in
						du2!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee) - 2) { du2 in
							cgttrf_(n, dl, d, du, du2, ipiv, info)
						}
					}
				}
			}
			return result
		} :
		{ n, dl, d, du, du2, ipiv, info in
			let result = dl!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee) - 1) { dl in
				d!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee)) { d in
					du!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee) - 1) { du in
						du2!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee) - 2) { du2 in
							zgttrf_(n, dl, d, du, du2, ipiv, info)
						}
					}
				}
			}
			return result
		}
	}
	
	public static var gttrs: gttrs<Complex<RealType>> {
		(RealType.self == Float.self ) ?
		{ trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info in
			let result = dl!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee) - 1) { dl in
				d!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee)) { d in
					du!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee) - 1) { du in
						du2!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee) - 2) { du2 in
							b!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(n!.pointee * nrhs!.pointee)) { b in
								cgttrs_(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info)
							}
						}
					}
				}
			}
			return result
		} :
		{ trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info in
			let result = dl!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee) - 1) { dl in
				d!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee)) { d in
					du!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee) - 1) { du in
						du2!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee) - 2) { du2 in
							b!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(n!.pointee * nrhs!.pointee)) { b in
								zgttrs_(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info)
							}
						}
					}
				}
			}
			return result
		}
	}
	
	public static var gtcon: gtcon<Complex<RealType>> {
		(RealType.self == Float.self) ?
		{ NORM, N, DL, D, DU, DU2, IPIV, ANORM, RCOND, WORK, iwork, INFO in
			DL!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(N!.pointee) - 1) { dl in
				D!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(N!.pointee)) { d in
					DU!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(N!.pointee) - 1) { du in
						DU2!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(N!.pointee) - 2) { du2 in
							ANORM!.withMemoryRebound(to: Float.self, capacity: 1) { anorm in
								RCOND!.withMemoryRebound(to: Float.self, capacity: 1) { rcond in
									WORK!.withMemoryRebound(to: __CLPK_complex.self, capacity: Int(2*N!.pointee)) { work in
										cgtcon_(NORM, N, dl, d, du, du2, IPIV, anorm, rcond, work, INFO)
									}
								}
							}
						}
					}
				}
			}
		} :
		{ NORM, N, DL, D, DU, DU2, IPIV, ANORM, RCOND, WORK, iwork, INFO in
			DL!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(N!.pointee) - 1) { dl in
				D!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(N!.pointee)) { d in
					DU!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(N!.pointee) - 1) { du in
						DU2!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(N!.pointee) - 2) { du2 in
							ANORM!.withMemoryRebound(to: Double.self, capacity: 1) { anorm in
								RCOND!.withMemoryRebound(to: Double.self, capacity: 1) { rcond in
									WORK!.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: Int(2*N!.pointee)) { work in
										zgtcon_(NORM, N, dl, d, du, du2, IPIV, anorm, rcond, work, INFO)
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	public static var axpy: axpy<Complex<RealType>> {
		(RealType.self == Float.self) ?
		{ n, a, x, incx, y, incy in
			withUnsafePointer(to: a) { aPtr in
				aPtr.withMemoryRebound(to: __CLPK_complex.self, capacity: 1) {
					cblas_caxpy(n, $0, x, incx, y, incy) } }
		} :
		{ n, a, x, incx, y, incy in
			withUnsafePointer(to: a) { aPtr in
				aPtr.withMemoryRebound(to: __CLPK_doublecomplex.self, capacity: 1) {
					cblas_zaxpy(n, $0, x, incx, y, incy) } }
		}
	}
	
}

// MARK: - Matrix Structures and Arithmetic

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
		size = diagonal.count
	}
	
	/// Calculates the 1-norm of the matrix (maximum absolute column sum).
	public func oneNorm() -> T.Magnitude {
		var norm: T.Magnitude = 0
		if size == 0 { return norm }
		if size == 1 { return diagonal[0].magnitude }
		
		// First column
		var colSum = diagonal[0].magnitude + lower[0].magnitude
		norm = colSum
		
		// Middle columns
		for j in 1..<(size - 1) {
			colSum = upper[j-1].magnitude + diagonal[j].magnitude + lower[j].magnitude
			norm = max(norm, colSum)
		}
		
		// Last column
		colSum = upper[size-2].magnitude + diagonal[size-1].magnitude
		norm = max(norm, colSum)
		
		return norm
	}
}

public typealias ColumnVector<T: ScalarField> = Array<T>

// ... ( `*` and `AXpY` functions remain unchanged) ...

public func *<T: ScalarField>(_ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>) -> ColumnVector<T> {
	precondition(x.count == A.size, "Invalid column vector size")
	let n = x.count
	var b : ColumnVector<T> = x
	if n==0 { return [] }
	b[0] = A.diagonal[0]*x[0]
	if n==1 { return b }
	b[0] += A.upper[0]*x[1]
	b[n-1] = A.lower[n-2]*x[n-2]+A.diagonal[n-1]*x[n-1]
	if n==2 { return b }
	for j in 1..<n-1 {
		b[j] = A.lower[j-1]*x[j-1] + A.diagonal[j]*x[j] + A.upper[j]*x[j+1]
	}
	return b
}

public func AXpY<T: ScalarField>(A: TridiagonalMatrix<T>, x: ColumnVector<T>, y: ColumnVector<T>) -> ColumnVector<T> {
	precondition(x.count == A.size, "Invalid x vector size")
	precondition(y.count == A.size, "Invalid y vector size")
	let n = x.count
	if n==0 { return [] }
	var b = y
	b[0] += A.diagonal[0]*x[0]
	if n==1 { return b }
	b[0] += A.upper[0]*x[1]
	b[n-1] += A.lower[n-2]*x[n-2]+A.diagonal[n-1]*x[n-1]
	if n==2 { return b }
	for j in 1..<n-1 {
		b[j] += A.lower[j-1]*x[j-1] + A.diagonal[j]*x[j] + A.upper[j]*x[j+1]
	}
	return b
}

// UPGRADED: This function now uses cblas_axpy via the ScalarField protocol.
public func aXpY<T: ScalarField>(a: T , x: ColumnVector<T>, y: ColumnVector<T>) -> ColumnVector<T> {
	precondition(x.count == y.count, "Vector size mismatch")
	
	let n = Int32(x.count)
	if n == 0 { return [] }
	
	var result = y
	T.axpy( n,  a,  x,  1,  &result,  1)
	return result
}

public struct TridiagonalLUMatrix<T: ScalarField> {
	
	// Stored LU factors and pivot vector
	public var subDiagonal: [T]
	public var mainDiagonal: [T]
	public var superDiagonal: [T]
	public var superDiagonal2: [T]
	public var ipiv: [__CLPK_integer]
	public var rcond: T.Magnitude
	/// The 1-norm of the original matrix A, computed before factorization.
	public var anorm: T.Magnitude
	
	public let count: Int
	public var approximateConditionNumber: T.Magnitude {  (rcond) > 0 ?  1/rcond : T.Magnitude.infinity }
	private var  det: T?
	public var determinant: T? { det }
	/// Private initializer to store factorization results.
	private init(subDiagonal: [T], mainDiagonal: [T], superDiagonal: [T], superDiagonal2: [T], ipiv: [__CLPK_integer],
				 rcond: T.Magnitude, anorm: T.Magnitude, determinant: T, count: Int) {
		self.subDiagonal = subDiagonal
		self.mainDiagonal = mainDiagonal
		self.superDiagonal = superDiagonal
		self.superDiagonal2 = superDiagonal2
		self.ipiv = ipiv
		self.rcond = rcond
		self.anorm = anorm
		self.count = count
		self.det = determinant
	}
	
	/// Convenience initializer that factors a `TridiagonalMatrix`.
	/// Returns `nil` if factorization fails (e.g., matrix is singular).
	public init?(_ A: TridiagonalMatrix<T>) {
		let anorm = A.oneNorm()
		self.init(dl: A.lower, d: A.diagonal, du: A.upper, anorm: anorm)
	}
	
	/// Factors a tridiagonal matrix defined by its diagonal arrays.
	/// Returns `nil` if factorization fails.
	private init?(dl: [T], d: [T], du: [T], anorm: T.Magnitude) {
		precondition(d.count > 1, "Matrix must be at least 2x2.")
		precondition(dl.count == d.count - 1 && du.count == d.count - 1, "Diagonal vector sizes are inconsistent.")
		
		let n = __CLPK_integer(d.count)
		var n_ = n
		var info = __CLPK_integer(0)
		var norm: Int8 = Int8(79) // 'O' for One-norm
		var infoFactor = __CLPK_integer(0)
		// LAPACK modifies these vectors in place
		var subDiagonal = dl
		var mainDiagonal = d
		var superDiagonal = du
		var superDiagonal2 = [T](repeating: T.zero, count: d.count - 1)
		var ipiv = [__CLPK_integer](repeating: 0, count: Int(n))
		var work = [T](repeating: 0, count: Int(2*n))
		var iwork = [CLPKInteger](repeating: 0, count: Int(2*n))
		var anorm_ = anorm
		var rcond_ = T.Magnitude(1)
		var determinant: T = 1
		
		subDiagonal.withUnsafeMutableBufferPointer { dl in let dlPtr = dl.baseAddress!
			mainDiagonal.withUnsafeMutableBufferPointer { d in let dPtr = d.baseAddress!
				superDiagonal.withUnsafeMutableBufferPointer { du in let duPtr = du.baseAddress!
					superDiagonal2.withUnsafeMutableBufferPointer { du2 in let du2Ptr = du2.baseAddress!
						ipiv.withUnsafeMutableBufferPointer { ipivBP in let ipivPtr = ipivBP.baseAddress!
							// Call the Generic Tridiagonal Factorization function
							_ = T.gttrf(&n_, dlPtr , dPtr, duPtr , du2Ptr, ipivPtr, &infoFactor)
							work.withUnsafeMutableBufferPointer { workBP in let workPtr = workBP.baseAddress!
								iwork.withUnsafeMutableBufferPointer { iworkBP in let iworkPtr = iworkBP.baseAddress!
									withUnsafeMutableBytes(of: &anorm_) { anormBytes in let anormPtr = anormBytes.baseAddress!
										withUnsafeMutableBytes(of: &rcond_) { rcondBytes in let rcondPtr = rcondBytes.baseAddress!
											_ = T.gtcon(&norm, &n_, dlPtr, dPtr, duPtr, du2Ptr, ipivPtr, anormPtr, rcondPtr,
														workPtr, iworkPtr, &info)
										}
									}
								}
							}
						}
					}
				}
			}
			
			guard infoFactor == 0 else { determinant = 0; return } // Factorization failed
			let detU = mainDiagonal.reduce(T.one, *)
			var sign = false
			for i in 0..<Int(n-1) { // n is CLPKInteger, last pivot is n
				if ipiv[i] != CLPKInteger(i + 1) { // 1-based indexing
					sign.toggle()
				}
			}
			determinant = sign ? -detU : detU
		}
		
		self.init(
			subDiagonal: subDiagonal,
			mainDiagonal: mainDiagonal,
			superDiagonal: superDiagonal,
			superDiagonal2: superDiagonal2,
			ipiv: ipiv,
			rcond: rcond_,
			anorm: anorm_,
			determinant: determinant,
			count: Int(n)
		)
		
	}
	// MARK: Solve and Condition
	
	/// Solves the tridiagonal system A * x = b using the pre-computed LU factors.
	@discardableResult
	public mutating func solve(rhsVector b: inout [T]) -> Bool {
	// While its not really mutating the call to gttrs needs mutating types so...
		precondition(b.count == self.count)
		
		var solutionVector = b
		
		var n_ = __CLPK_integer(self.count)
		var nrhs = __CLPK_integer(1)
		var ldb = n_
		var info = __CLPK_integer(0)
		var trans = Int8(78) // 'N'
		//var result: __CLPK_integer = 0
		
		self.subDiagonal.withUnsafeMutableBufferPointer { dl in let dlPtr = dl.baseAddress!
			self.mainDiagonal.withUnsafeMutableBufferPointer { d in let dPtr = d.baseAddress!
				self.superDiagonal.withUnsafeMutableBufferPointer { du in let duPtr = du.baseAddress!
					self.superDiagonal2.withUnsafeMutableBufferPointer { du2 in let du2Ptr = du2.baseAddress!
						self.ipiv.withUnsafeMutableBufferPointer { ipivIn in let ipivPtr = ipivIn.baseAddress!
							solutionVector.withUnsafeMutableBufferPointer { b in let bPtr = b.baseAddress!
								_ = T.gttrs(&trans, &n_, &nrhs, dlPtr, dPtr, duPtr, du2Ptr, ipivPtr, bPtr, &ldb, &info)
							}
						}
					}
				}
			}
		}
		
		guard info == 0 else { return false }
		
		b = solutionVector
		
		return true
	}
	
}

// MARK: - Usage Examples

/*
 // Example 1: Real Double Precision (T = Double)
 
 let A_D = TridiagonalMatrix(diagonal: [2.0, 5.0, 4.0],
 upper: [1.0, 1.0],
 lower: [3.0, 3.0])
 
 // D. Factor the matrix using the new initializer
 guard let factoredA_D = TridiagonalLUMatrix(from: A_D) else {
 fatalError("Double Factorization failed.")
 }
 
 // F. Get the condition number
 print("Condition Number (Double): \(factoredA_D.conditionNumber)")
 
 // E. Solve the system A*x = b
 var b_solve: [Double] = [4.0, 10.0, 7.0]
 if factoredA_D.solve(rhsVector: &b_solve) {
 print("Real Double Solution x: \(b_solve)")
 }
 
 // Example 2: Complex Float Precision (T = Numerics.Complex<Float>)
 typealias CFloat = Complex<Float>
 
 let A_CF = TridiagonalMatrix(diagonal: [CFloat(1.0, -1.0), CFloat(3.0, -1.0)],
 upper: [CFloat(2.0, 0.0)],
 lower: [CFloat(1.0, 0.0)])
 
 // C. Factor and solve
 guard let factoredA_CF = TridiagonalLUMatrix(from: A_CF) else {
 fatalError("Complex Float Factorization failed.")
 }
 
 // F. Get condition number
 print("Condition Number (Complex): \(factoredA_CF.conditionNumber)")
 
 // D. Solve
 var b_solve_CF: [CFloat] = [CFloat(4.0, 0.0), CFloat(4.0, 0.0)]
 if factoredA_CF.solve(rhsVector: &b_solve_CF) {
 print("Complex Float Solution x: \(b_solve_CF)")
 }
 
 */
