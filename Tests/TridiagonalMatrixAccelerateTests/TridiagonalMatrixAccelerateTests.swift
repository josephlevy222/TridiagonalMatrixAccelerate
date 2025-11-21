import XCTest
@testable import TridiagonalMatrixAccelerate
import Numerics
import Accelerate

final class TridiagonalMatrixAccelerateTests: XCTestCase {
	/// Asserts that two vectors are approximately equal within a given tolerance.
	private func assertVectorsApproximatelyEqual<T: ScalarField>(_ v1: [T], _ v2: [T],
																 tolerance: T.Magnitude,
																 file: StaticString = #file,
																 line: UInt = #line) {
		XCTAssertEqual(v1.count, v2.count, "Vector counts do not match", file: (file), line: line)
		for (i, (a, b)) in zip(v1, v2).enumerated() {
			let error = (a - b).magnitude
			XCTAssertTrue(error < tolerance, "Vector element \(i) not equal. (\(a) vs \(b), error \(error))",
						  file: (file), line: line)
		}
	}
	
	func testNormal() throws {
		try testExample(Complex<Double>(2.0,0.0), det: Complex<Double>(6.0,0.0))
	}
	func testSingular() throws {
		try testExample(Complex<Float>(1.0,0.0), det: Complex<Float>(0.0,0.0))
	}
	func testNearlySingular() throws {
		try testExample(Float(1.732051), det: Float(2.02656e-6))
	}
	
	func testaXpY() {
		// Test with Double
		let a_d: Double = 2.0
		let x_d: [Double] = [1.0, 2.0, 3.0]
		let y_d: [Double] = [10.0, 20.0, 30.0]
		
		let result_d = aXpY(a: a_d, x: x_d, y: y_d)
		let expected_d: [Double] = [12.0, 24.0, 36.0]
		
		assertVectorsApproximatelyEqual(result_d, expected_d, tolerance: 1e-15)
		
		// Test with Complex<Float>
		typealias CFloat = Complex<Double>
		let a_c = CFloat(1.0, 1.0) // (1+i)
		let x_c = [CFloat(2.0, 0.0), CFloat(0.0, 3.0)] // [2, 3i]
		let y_c = [CFloat(10.0, 0.0), CFloat(20.0, 0.0)] // [10, 20]
		
		// Expected calculation:
		// a*x = [(1+i)*2, (1+i)*3i] = [2+2i, 3i-3] = [2+2i, -3+3i]
		// a*x + y = [12+2i, 17+3i]
		
		let result_c = aXpY(a: a_c, x: x_c, y: y_c)
		let expected_c = [CFloat(12.0, 2.0), CFloat(17.0, 3.0)]
		
		assertVectorsApproximatelyEqual(result_c, expected_c, tolerance: 1e-7)
	}
	
	func testMatrixVectorMultiply() {
		// Test with Double
		let A = TridiagonalMatrix(diagonal: [2.0, 3.0, 4.0],
								  upper: [1.0, 1.0],
								  lower: [-1.0, -1.0])
		let x: [Double] = [1.0, 2.0, 3.0]
		
		// Expected calculation:
		// A*x = | 2  1  0 | | 1 |   | 2*1 + 1*2 + 0*3 |   |  4 |
		//       |-1  3  1 | | 2 | = | -1*1 + 3*2 + 1*3| = |  8 |
		//       | 0 -1  4 | | 3 |   | 0*1 + -1*2 + 4*3|   | 10 |
		
		let result = A * x
		let expected: [Double] = [4.0, 8.0, 10.0]
		
		assertVectorsApproximatelyEqual(result, expected, tolerance: 1e-15)
	}
	
	func testAXpY() {
		// Test with Double
		let A = TridiagonalMatrix(diagonal: [2.0, 3.0, 4.0],
								  upper: [1.0, 1.0],
								  lower: [-1.0, -1.0])
		let x: [Double] = [1.0, 2.0, 3.0]
		let y: [Double] = [100.0, 200.0, 300.0]
		
		// Expected calculation:
		// A*x = [4.0, 8.0, 10.0] (from testMatrixVectorMultiply)
		// A*x + y = [104.0, 208.0, 310.0]
		
		let result = AXpY(A: A, x: x, y: y)
		let expected: [Double] = [104.0, 208.0, 310.0]
		
		assertVectorsApproximatelyEqual(result, expected, tolerance: 1e-15)
	}
	
	private func testExample<T : ScalarField >(_ d: T, det: T)  throws {
		let one = T.one
		let zero = T.zero
		let lower = [one,one,one,one]
		let upper = lower
		let diagonal = [d,d,d,d,d]
		let tridiag = TridiagonalMatrix(diagonal: diagonal, upper: upper, lower: lower)
		let tridiagLU =  TridiagonalLUMatrix(tridiag)
		let i = [ [one,zero,zero,zero,zero],
				  [zero,one,zero,zero,zero],
				  [zero,zero,one,zero,zero],
				  [zero,zero,zero,one,zero],
				  [zero,zero,zero,zero,one] ]
		let x = i.map {icol in var b = icol; return  tridiagLU.solve(&b) }
		let ii = x.map {xcol in tridiag*xcol}
		var tolerance = T.Magnitude.ulpOfOne*2
		let condition = tridiagLU.approximateConditionNumber
		print("conditionNumber=\(condition)")
		tolerance *= condition
		print("tolerance=\(tolerance)")
		let e = zip(i,ii).map { zip($0,$1).map { $0 - $1 } }
		let okay = e.flatMap { $0 }.reduce(true) { $0 && $1.magnitude < tolerance }
		let maxError = e.flatMap {$0}.reduce(T.Magnitude.zero) { max($0,$1.magnitude)}
		print("maxError=\(maxError)")
		let determinant = tridiagLU.determinant
		print("determinate=\(determinant) vs. \(det)")
		//XCTAssertTrue(determinant.isApproximatelyEqual(to: det))
		// FIX 2: Replace `isApproximatelyEqual` with a magnitude check
		let detError = (determinant - det).magnitude
		// Use an absolute tolerance for determinant check
		let detTolerance = T.Magnitude.ulpOfOne * 1000
		
		XCTAssertTrue(detError < detTolerance, "Determinant error \(detError) exceeds tolerance \(detTolerance)")
		XCTAssertTrue(okay || tridiagLU.approximateConditionNumber.isInfinite )
	}
}


