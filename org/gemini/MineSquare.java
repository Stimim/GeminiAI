package org.gemini;

import com.google.gwt.user.client.rpc.IsSerializable;

/**
 * @author stimim
 * 
 * if a square is an instance of MineSquare, there must be a mine.
 */
public class MineSquare extends KnownSquare implements IsSerializable {
	public static MineSquare FoundMine = null ;
	/**
	 * -1 : player
	 *  0 : nobody
	 *  1 : AI (me)
	 */
	public int owner ;
	public MineSquare(int x, int y, int owner) {
		super(x, y);
		// TODO Auto-generated constructor stub
		this.owner = owner ;
		
		if( this.owner == 0 ) {
			MineSquare.FoundMine = this ;
		}
	}
}
