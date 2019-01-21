caching:
	$(MAKE) -C caching_classes all
	$(MAKE) -C caching_classes copy

caching_clean:
	rm environment/*.so
	rm environment/*.o
	$(MAKE) -C caching_classes clean



collector:	
	$(MAKE) -C feature_collector all

collector_clean:
	$(MAKE) -C feature_collector clean
	
all:
	make caching
	make collector

clean:
	make caching_clean
	make collector_clean
