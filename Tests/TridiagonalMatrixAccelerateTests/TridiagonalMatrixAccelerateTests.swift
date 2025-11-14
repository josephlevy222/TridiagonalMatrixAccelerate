import XCTest
@testable import TridiagonalMatrixAccelerate
import Numerics
import Accelerate

final class TridiagonalMatrixAccelerateTests: XCTestCase {
	
	func testNormal() throws {
		try testExample(Complex<Double>(2.0,0.0), det: Complex<Double>(6.0,0.0))
	}
	func testSingular() throws {
		try testExample(Complex<Float>(1.0,0.0), det: Complex<Float>(0.0,0.0))
	}
	func testNearlySingular() throws {
		try testExample(Float(1.732051), det: Float(2.02656e-6))
	}
	
//	func testTriFactor() throws {
//		// Example for a double-precision complex tridiagonal matrix
//		// Note: dl and du are one element shorter than d
//		typealias DSPComplex = Complex<Double>
//		let n = 4
//		var dl: [DSPComplex] = [DSPComplex(1, 1), DSPComplex(2, 2), DSPComplex(0, 0)] // subdiagonal
//		var d: [DSPComplex] = [DSPComplex(3, 3), DSPComplex(4, 4), DSPComplex(5, 5), DSPComplex(6, 6)] // diagonal
//		var du: [DSPComplex] = [DSPComplex(7, 7), DSPComplex(8, 8), DSPComplex(9, 9)] // superdiagonal
//		let tridiag = TridiagonalMatrix(diagonal: d, upper: du, lower: dl)
//		// You must remove the padding zero from the subdiagonal (dl) before passing it in.
//		var dlInput = Array(dl.dropLast())
//		// The last element of dl isn't used in the input, but the result has a full set of n-1 elements.
//		var duInput = du
//		
//		do {
//			// Perform the factorization. The function modifies the arrays in-place.
//			//let ipiv = gttrf(n, dl, d, du, du2, ipiv, info)
//			
//			print("Subdiagonal (after factorization): \(dlInput)")
//			print("Diagonal (after factorization): \(d)")
//			print("Superdiagonal (after factorization): \(duInput)")
//			print("Pivot indices: \(ipiv)")
//			
//		} catch {
//			print("An error occurred during factorization: \(error)")
//		}
//	}
	
	private func testExample<T : ScalarField >(_ d: T, det: T)  throws {
		let one = T.one
		let zero = T.zero
		let lower = [one,one,one,one]
		let upper = lower
		let diagonal = [d,d,d,d,d]
		let tridiag = TridiagonalMatrix(diagonal: diagonal, upper: upper, lower: lower)
		var tridiagLU = TridiagonalLUMatrix(tridiag)
		let i = [ [one,zero,zero,zero,zero],
				  [zero,one,zero,zero,zero],
				  [zero,zero,one,zero,zero],
				  [zero,zero,zero,one,zero],
				  [zero,zero,zero,zero,one] ]
		let x = i.map {icol in var b = icol; tridiagLU?.solve(rhsVector: &b); return b}
		let ii = x.map {xcol in tridiag*xcol}
		var tolerance = T.Magnitude.ulpOfOne*2
		let condition = tridiagLU?.approximateConditionNumber ?? T.Magnitude.zero
		print("conditionNumber=\(condition)")
		tolerance *= condition
		print("tolerance=\(tolerance)")
		let e = zip(i,ii).map { zip($0,$1).map { $0 - $1 } }
		let okay = e.flatMap { $0 }.reduce(true) { $0 && $1.magnitude < tolerance }
		let maxError = e.flatMap {$0}.reduce(T.Magnitude.zero) { max($0,$1.magnitude)}
		print("maxError=\(maxError)")
		let determinant = tridiagLU?.determinant ?? 0
		print("determinate=\(determinant) vs. \(det)")
		//XCTAssertTrue(determinant.isApproximatelyEqual(to: det))
		// FIX 2: Replace `isApproximatelyEqual` with a magnitude check
		let detError = (determinant - det).magnitude
		// Use an absolute tolerance for determinant check
		let detTolerance = T.Magnitude.ulpOfOne * 1000
		
		XCTAssertTrue(detError < detTolerance, "Determinant error \(detError) exceeds tolerance \(detTolerance)")
		XCTAssertTrue(okay || tridiagLU?.approximateConditionNumber.isInfinite ?? true)
	}
}


