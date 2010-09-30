package org.gemini;

import com.google.gwt.user.client.rpc.IsSerializable;

class Square implements IsSerializable {
	public int x;
	public int y;
	
	public Square( int x_ , int y_ )
	{
		this.x = x_ ;
		this.y = y_ ;
	}
}
