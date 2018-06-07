from torch import nn

from util import log


class Network ( nn.Module ):

	def __init__ ( self ):
		super ( Network, self ).__init__ ()

		self.conv1 = nn.Sequential (
				nn.Conv2d ( in_channels = 1, out_channels = 4, kernel_size = 3 ),
				nn.BatchNorm2d ( num_features = 4 ) )
		self.conv2 = nn.Sequential (
				nn.Conv2d ( in_channels = 4, out_channels = 16, kernel_size = 3 ),
				nn.BatchNorm2d ( num_features = 16 )
		)
		self.pl3 = nn.Sequential (
				nn.MaxPool2d ( kernel_size = 3 ),
				nn.BatchNorm2d ( num_features = 16 )
		)
		self.conv4 = nn.Sequential (
				nn.Conv2d ( in_channels = 16, out_channels = 4, kernel_size = 3 ),
				nn.BatchNorm2d ( num_features = 4 )
		)
		self.conv5 = nn.Sequential (
				nn.Conv2d ( in_channels = 4, out_channels = 1, kernel_size = 1 ),
				nn.BatchNorm2d ( num_features = 1 )
		)
		self.fc7 = nn.Sequential (
				nn.Linear ( in_features = 6 * 6, out_features = 32 ),
				nn.Dropout (),
				nn.ELU ()
		)
		self.fc8 = nn.Sequential (
				nn.Linear ( in_features = 32, out_features = 10 )
		)

	def forward ( self, x ):
		log ( str ( x.size () ) )
		log ( "Running through network!" )
		x = self.conv1 ( x )
		log ( str ( x.size () ) )
		log ( "Went through conv1" )
		x = self.conv2 ( x )
		log ( str ( x.size () ) )
		log ( "Went through conv2" )
		x = self.pl3 ( x )
		log ( str ( x.size () ) )
		log ( "Went through pl3" )
		x = self.conv4 ( x )
		log ( str ( x.size () ) )
		log ( "Went through conv4" )
		x = self.conv5 ( x )
		log ( str ( x.size () ) )
		log ( "Went through conv5" )
		x = x.view ( -1, 6 ** 2 )
		log ( str ( x.size () ) )
		x = self.fc7 ( x )
		log ( "Went through fc7" )
		log ( "Returning output in a bit" )
		return self.fc8 ( x )
