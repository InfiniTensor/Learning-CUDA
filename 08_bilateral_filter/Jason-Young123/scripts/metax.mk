define PLATFORM_INIT_ENV
true
endef

exec: ./runTester
	export MACA_DEVICE_IMAGE_CHECK=1
	./runTester
