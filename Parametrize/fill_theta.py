import random
def FillTheta(theta_arr,masses,excl_mass=[]):
	if excl_mass:
		for i in range(len(excl_mass)):
			masses.remove(excl_mass[i])
	nmasses=len(masses)
	for i in range(len(theta_arr)):
		theta_arr[i]=masses[random.randint(0,nmasses-1)]
	return theta_arr


prova=[0,0,0,0,0,0,0,0,0]
masses=[10,20,30]

print(FillTheta(prova,masses,[30,20]))