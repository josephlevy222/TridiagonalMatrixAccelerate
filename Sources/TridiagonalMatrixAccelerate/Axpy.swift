//
//  Axpy.swift
//  TridiagonalMatrixAccelerate
//
//  Created by Joseph Levy on 12/3/25.
//
import Accelerate
public func originalAXpY<T: ScalarField>(A: TridiagonalMatrix<T>, x: ColumnVector<T>, y: ColumnVector<T>) -> ColumnVector<T> {
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

import Numerics

@inlinable public func AXpY_<T: RealScalar>( _ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>, _ y: inout ColumnVector<T>
) -> ColumnVector<T> {
	let n = x.count
	precondition(n == A.size)
	
	if n == 0 { return [] }
	
	if n == 1 {
		y[0] = A.diagonal[0] * x[0]
		return y
	}
	
	A.diagonal.withUnsafeBufferPointer { dPtr in
		x.withUnsafeBufferPointer { xPtr in
			y.withUnsafeMutableBufferPointer { yPtr in
				
				// y = d * x
				T.vmul(dPtr.baseAddress!, 1, xPtr.baseAddress!, 1, yPtr.baseAddress!, 1, n )
				
				// y[0...n-2] += upper * x[1...n-1]
				A.upper.withUnsafeBufferPointer { uPtr in
					T.vma(uPtr.baseAddress!, 1, xPtr.baseAddress! + 1, 1, yPtr.baseAddress!, 1, n - 1 )
				}
				
				// y[1...n-1] += lower * x[0...n-2]
				A.lower.withUnsafeBufferPointer { lPtr in
					T.vma( lPtr.baseAddress!, 1, xPtr.baseAddress!, 1, yPtr.baseAddress! + 1, 1, n - 1 )
				}
			}
		}
	}
	return y
}

/// Helper: Complex multiply-add for a band: y += band * x
@inline(__always) public func complexBandMA<T: RealScalar>(
	_ band: UnsafePointer<T>, _ x: UnsafePointer<T>, _ y: CMutablePtr<T>, _ temp: CMutablePtr<T>, _ count: Int) {
	T.vma(band, 2, x, 2, y, 2, count)          // y.real += d.real * x.real
	T.vmul(band+1, 2, x+1, 2, temp, 1, count)  // y.real += d.real * x.real
	T.vsub(temp, 1, y, 2, y, 2, count)         // y.real = y.real - temp
	T.vma(band, 2, x+1, 2, y+1, 2, count)      // y.imag += d.real * x.imag
	T.vma(band+1, 2, x, 2, y+1, 2, count)      // y.imag += d.imag * x.real
}

@inlinable public func AXpY_<T: RealScalar>(
	_ A: TridiagonalMatrix<Complex<T>>, _ x: [Complex<T>], _ y: inout [Complex<T>]
) -> [Complex<T>] {
	
	let n = x.count
	precondition(n == A.size)
	precondition(y.count == n)
	
	if n == 0 { return y }
	
	if n == 1 {
		y[0] = y[0] + A.diagonal[0] * x[0]
		return y
	}
	
	// We need temp of size of diagonal = n, but off-diagonals only use n-1
	let tempSize = n  // For diagonal computations
	
	if tempSize <= 1024 {
		// Small: Use stack for both
		withUnsafeTemporaryAllocation(of: T.self, capacity: tempSize) { diagTemp in
			computeAXpYWithTemps(A, x: x, y: &y, diagTemp: diagTemp.baseAddress!)
		}
	} else {
		// Large: Use heap
		let diagTemp = UnsafeMutableBufferPointer<T>.allocate(capacity: tempSize)
		defer { diagTemp.deallocate() }
		computeAXpYWithTemps(A, x: x, y: &y, diagTemp: diagTemp.baseAddress!)
	}
	return y
}

// Version that reuses same temp array for both diagonal and off-diagonal
@inlinable internal func computeAXpYWithTemps<T: RealScalar>(
	_ A: TridiagonalMatrix<Complex<T>>, x: [Complex<T>], y: inout [Complex<T>],  diagTemp: UnsafeMutablePointer<T>
) {
	let n = x.count
	
	x.withUnsafeBufferPointer { xPtr in
		y.withUnsafeMutableBufferPointer { yPtr in
			
			let xBase = xPtr.baseAddress!
			let yBase = yPtr.baseAddress!
			
			xBase.withMemoryRebound(to: T.self, capacity: 2*n) { x in
				yBase.withMemoryRebound(to: T.self, capacity: 2*n) { y in
					// --- DIAGONAL ---
					A.diagonal.withUnsafeBufferPointer { dPtr in
						let dBase = dPtr.baseAddress!
						dBase.withMemoryRebound(to: T.self, capacity: 2*n) { d in
							complexBandMA(d, x, y, diagTemp, n)
						}
					}
					// --- UPPER DIAGONAL ---
					A.upper.withUnsafeBufferPointer { uPtr in
						let uBase = uPtr.baseAddress!
						uBase.withMemoryRebound(to: T.self, capacity: 2*(n-1)) { upper in
							complexBandMA(upper, x+2, y, diagTemp, n-1)
						}
					}
					
					// --- LOWER DIAGONAL ---
					A.lower.withUnsafeBufferPointer { lPtr in
						let lBase = lPtr.baseAddress!
						lBase.withMemoryRebound(to: T.self, capacity: 2*(n-1)) { lower in
							complexBandMA(lower, x, y+2, diagTemp, n-1)
						}
					}
				}
			}
			
		}
	}
}

// MARK: - Tridiagonal Matrix-Vector Multiplication
public func *<T: RealScalar>(_ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>) -> ColumnVector<T> {
	precondition(x.count == A.size, "Invalid column vector size")
	let n = x.count
	if n == 0 { return [] }
	if n == 1 { return [A.diagonal[0] * x[0]] }
	
	var result = [T](unsafeUninitializedCapacity: n) { buffer, initializedCount in
		initializedCount = n
	}
	
	x.withUnsafeBufferPointer { xPtr in
		result.withUnsafeMutableBufferPointer { resultPtr in
			let xBase = xPtr.baseAddress!
			let r = resultPtr.baseAddress!
			// Diagonal contribution: result = diagonal * x
			A.diagonal.withUnsafeBufferPointer { dPtr in
				T.vmul(dPtr.baseAddress!, 1, xBase, 1, r, 1, n)
			}
			// Upper diagonal: result[0..n-2] += upper * x[1..n-1]
			A.upper.withUnsafeBufferPointer { uPtr in
				T.vma(uPtr.baseAddress!, 1, xBase + 1, 1, r, 1, n-1)
			}
			// Lower diagonal: result[1..n-1] += lower * x[0..n-2]
			A.lower.withUnsafeBufferPointer { lPtr in
				T.vma(lPtr.baseAddress!, 1, xBase, 1, r + 1, 1, n-1)
			}
		}
		
	}
	
	return result
}

public func *<T: RealScalar >(_ A: TridiagonalMatrix<Complex<T>>, _ x: ColumnVector<Complex<T>>) -> ColumnVector<Complex<T>> {
	precondition(x.count == A.size, "Invalid column vector size")
	let n = x.count
	if n == 0 { return [] }
	if n == 1 { return [A.diagonal[0] * x[0]] }
	
	var result = [Complex<T>](unsafeUninitializedCapacity: n) { buffer, initializedCount in
		initializedCount = n
	}
	
	var temp = [T](unsafeUninitializedCapacity: n) { buffer, initializedCount in
		initializedCount = n
	}
	
	x.withUnsafeBufferPointer { xPtr in
		result.withUnsafeMutableBufferPointer { resultPtr in
			
			let xBase = xPtr.baseAddress!
			let resultBase = resultPtr.baseAddress!
			
			xBase.withMemoryRebound(to: T.self, capacity: 2*n) { x in
				resultBase.withMemoryRebound(to: T.self, capacity: 2*n) { y in
					
					// --- DIAGONAL: result = diagonal * x ---
					A.diagonal.withUnsafeBufferPointer { dPtr in
						let dBase = dPtr.baseAddress!
						dBase.withMemoryRebound(to: T.self, capacity: 2*n) { d in
							// Real part: result.real = d.real * x.real - d.imag * x.imag
							T.vmul(d, 1, x, 1, y, 1, 2*n) // Use stride-1 trick to get both products at once
							T.vsub(y+1, 2, y, 2, y, 2, n) //result[even] = result[even] - result[odd]
							// Imaginary part: result.imag = d.real * x.imag + d.imag * x.real
							T.vmul(d, 2, x+1, 2, y+1, 2, n)
							T.vma(d+1, 2, x, 2, y+1, 2, n)
						}
					}
					
					// --- UPPER DIAGONAL: result[0..n-2] += upper * x[1..n-1] ---
					A.upper.withUnsafeBufferPointer { uPtr in
						let uBase = uPtr.baseAddress!
						uBase.withMemoryRebound(to: T.self, capacity: 2*(n-1)) { upper in
							complexBandMA(upper, x+2, y, &temp, n-1)
						}
					}
					
					// --- LOWER DIAGONAL: result[1..n-1] += lower * x[0..n-2] ---
					A.lower.withUnsafeBufferPointer { lPtr in
						let lBase = lPtr.baseAddress!
						lBase.withMemoryRebound(to: T.self, capacity: 2*(n-1)) { lower in
							complexBandMA(lower, x, y+2, &temp, n-1)
						}
					}
				}
			}
		}
	}
	return result
}
