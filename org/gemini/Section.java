package org.gemini;

import java.util.HashSet;

@SuppressWarnings("serial")
public class Section extends HashSet<NumberedSquare> {
	public RecordSet recordSet ;
	public Section() {
		super() ;
		recordSet = new RecordSet() ;
	}
}
