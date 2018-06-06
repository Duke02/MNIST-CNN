from torch import nn


class Network ( nn.Module ):

	def __init__ ( self ):
		super ( Network, self ).__init__ ()

		self.conv1 = nn.Sequential (
			nn.Conv2d ( in_channels = 1, out_channels = 4, kernel_size = 5 ),
			nn.BatchNorm2d ( num_features = 16 ) )
		self.conv2 = nn.Sequential (
			nn.Conv2d ( in_channels = 4, out_channels = 16, kernel_size = 3 ),
			nn.BatchNorm2d ( num_features = 16 )
		)
		self.conv3 = nn.Sequential (
			nn.Conv2d ( in_channels = 16, out_channels = 16, kernel_size = 5, stride = 2 ),
			nn.BatchNorm2d ( num_features = 16 )
		)
		self.pl4 = nn.Sequential (
			nn.MaxPool2d ( kernel_size = 3 ),
			nn.BatchNorm2d ( num_features = 16 )
		)
		self.conv5 = nn.Sequential (
			nn.Conv2d ( in_channels = 16, out_channels = 4, kernel_size = 3 ),
			nn.BatchNorm2d ( num_features = 4 )
		)
		self.conv6 = nn.Sequential (
			nn.Conv2d ( in_channels = 4, out_channels = 1, kernel_size = 3 ),
			nn.BatchNorm2d ( num_features = 1 )
		)
		self.pl7 = nn.Sequential (
			nn.AvgPool2d ( kernel_size = 3 ),
			nn.BatchNorm2d ( num_features = 1 )
		)
		self.fc8 = nn.Sequential (
			nn.Linear ( in_features = 379, out_features = 32 ),
			nn.ReLU ()
		)
		self.fc9 = nn.Sequential (
			nn.Linear ( in_features = 32, out_features = 10 ),
			nn.ReLU ()
		)

	def forward ( self, x ):
		x = self.conv1 ( x )
		x = self.conv2 ( x )
		x = self.conv3 ( x )
		x = self.pl4 ( x )
		x = self.conv5 ( x )
		x = self.conv6 ( x )
		x = self.pl7 ( x )
		x = self.fc8 ( x )
		return self.fc9 ( x )
