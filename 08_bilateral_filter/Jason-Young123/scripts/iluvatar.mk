define PLATFORM_INIT_ENV
if [ "$$(id -u)" -eq 0 ]; then \
    apt update && apt install libopencv-dev -y; \
else \
    true; \
fi
endef

exec: ./runTester
	./runTester
