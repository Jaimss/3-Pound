# code to determine if a person is healthy or not
age = int(input('What is the persons age? '))

if (age > 40):
    # did or do they smoke
    smoke = input('Do they smoke? (y/N)')
    if (smoke == 'y'):
        print('unhealthy')
    else:
        print('healthy')
    pass
else:
    # do they eat a lot of junk food
    junk_food = input('Do they eat a lot of junk food? (y/N)')
    if (junk_food == 'y'):
        print('unhealthy')
    else:
        print('healthy')

