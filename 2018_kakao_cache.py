from collections import deque

def solution(cacheSize, cities):
    cache = deque()
    answer = 0

    if cacheSize == 0:
        return 5 * len(cities)

    for city in cities:
        city = city.lower()

        if city in cache:
            answer += 1
            cache.remove(city)
            cache.append(city)

        else:
            answer += 5

            if len(cache) >= cacheSize:
                cache.popleft()
                cache.append(city)
            else:
                cache.append(city)

    return answer