caching:
	$(MAKE) -C caching_classes all
	$(MAKE) -C caching_classes copy

caching_clean:
	rm environment/*.so
	rm environment/*.o
	$(MAKE) -C caching_classes clean



feature_collector:
	$(MAKE) -C feature_collector all

feature_collector_clean:
	$(MAKE) -C feature_collector clean

reward_collector:
	$(MAKE) -C reward_collector all

reward_collector_clean:
	$(MAKE) -C reward_collector clean


all:
	make caching
	make feature_collector
	make reward_collector

clean:
	make caching_clean
	make feature_collector_clean
	make reward_collector_clean
