from datetime import datetime

shouldLog = True


def log ( message: str ):
	if shouldLog:
		print ( str ( datetime.now () ) + message )
