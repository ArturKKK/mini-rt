# Создаём отдельный бинарь
add_executable(minirt-test minirt_test.cpp)

# Линкуем библиотеку проекта
target_link_libraries(minirt-test PUBLIC minirt)

# Подключаем OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(minirt-test PUBLIC OpenMP::OpenMP_CXX)
