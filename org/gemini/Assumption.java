package org.gemini;

import java.util.Set;

import com.google.gwt.user.client.rpc.IsSerializable;

public class Assumption implements IsSerializable {
	public UnknownSquare[] array;
	public int[] assumedMine;
	public int remainder;
	public int arraySize;

	public Assumption(int remainder) {
		this.remainder = remainder;
		this.arraySize = 0;
		this.assumedMine = null;
		this.array = null;
	}

	public boolean initialize(Set<UnknownSquare> set) {
		for (UnknownSquare v : set) {
			switch (v.assumed) {
			case 0: // is not assumed yet
				arraySize++;
				break;
			case 1: // mine
				remainder--;
				break;
			default:
			case 2: // zero
				break;
			}
		}

		if (arraySize < remainder)
			return false;

		if (remainder < 0)
			return false;

		array = new UnknownSquare[arraySize];
		assumedMine = new int[remainder + 1];

		int i = 0;
		for (UnknownSquare v : set) {
			if (v.assumed == 0) {
				array[i++] = v;
			}
		}

		for (i = 0; i < remainder; i++) {
			assumedMine[i] = i;
		}

		assumedMine[remainder] = arraySize;

		return true;
	}

	public boolean nextPermutation() {
		if (remainder == 0)
			return false;
		for (int i = 0; i < remainder; i++) {
			if (assumedMine[i] + 1 < assumedMine[i + 1]) {
				assumedMine[i]++;
				break;
			} else {
				assumedMine[i] = i;
				if (i == remainder - 1)
					return false;
			}
		}
		return true;
	}

	public void applyAssumption() {
		for (int i = 0; i < arraySize; i++)
			array[i].assumed = 2;
		for (int i = 0; i < remainder; i++)
			array[assumedMine[i]].assumed = 1;
	}

	public void restore() {
		if( array == null )
			return ;
		for (int i = 0; i < arraySize; i++)
			array[i].assumed = 0;
	}
}
