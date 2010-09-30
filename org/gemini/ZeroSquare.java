package org.gemini;

import com.google.gwt.user.client.rpc.IsSerializable;

public class ZeroSquare extends KnownSquare implements IsSerializable {
	/**
	 * false: 原地圖上是 Zero 
	 * true : 原地圖上是 Unknown
	 */
	public boolean iFound;

	public ZeroSquare(int x, int y) {
		super(x, y);
		this.iFound = false ;
	}

	public ZeroSquare(int x, int y, boolean flag) {
		super(x, y);
		// TODO Auto-generated constructor stub
		this.iFound = flag;
	}
}
