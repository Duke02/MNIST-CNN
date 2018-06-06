from datetime import datetime

shouldLog = False


def log ( message: str ):
	if shouldLog:
		print ( str ( datetime.now () ) + message )
