#cmd ukaz iz mape, kjer je ta skripta: python3 -m cProfile -s time edit_video.py > D:\AnonAI\anonai-skupina1\prfofiling\profile.txt
#cmd ukaz iz mape, kjer je ta skripta: python3 -m cProfile -o profiling\edit_video.cprof edit_video.py
# #                                    python -m cProfile [-o output_file] [-s sort_order] profiling\edit_video.py

##tottime
##for the total time spent in the given function (and excluding time made in calls to sub-functions)

##cumtime
##is the cumulative time spent in this and all subfunctions (from invocation till exit).
##This figure is accurate even for recursive functions.

def read_file(file_name):
    with open(file_name, "r") as file:
        start = 0
        head = False
        
        for line in file:
            if line.strip().startswith("Ordered by"):
                start = 1
                
            elif start == 1:
                line = line.strip().split()
                if line != []:
                  
                    if head == False:
                        head = line
                        head[-2] = "percall2"
                        #print(head)
                        profile = {key: [] for key in line}
                    else:
                        for i in range(0, len(head)-1):
                            if "/" not in line[i]:
                                profile[head[i]].append(float(line[i]))
                            else:
                                #print(line[i])
                                #print(float(line[i].split("/")[0])/float(line[i].split("/")[1]))
                                profile[head[i]].append(float(line[i].split("/")[0])/float(line[i].split("/")[1]))
                            
                        profile[head[-1]].append("".join(line[5:]))

                    
    return profile

profile = read_file("profiling/profile_rabbit2_2.txt")

##kr neki
##print(profile.keys())
##print(profile["percall"][0:5])
##
##percall_tuples = list(enumerate(profile["percall"])) # [(0, lala), (1, bb), ..]
##
##print(max(list(map(lambda x: x[1], percall_tuples)))) # edit video 0.25
##
##percall_tuples.sort(key = lambda x:x[1], reverse = True) #[(1, bb), (0, lala), ..]
##print(percall_tuples[0:5])
##percall = list(map(lambda x:x[1], percall_tuples))
##percall_indexes = list(map(lambda x:x[0], percall_tuples))
##
##function_names1 = profile['filename:lineno(function)']
##function_names2 = list(zip(percall_indexes, function_names1))
##function_names3 = sorted(function_names2)


percall_and_name = list(zip(profile["percall"], profile['filename:lineno(function)']))
percall_and_name.sort(reverse = True)


tottime_and_name = list(zip(profile["tottime"], profile['filename:lineno(function)']))
tottime_and_name.sort(reverse = True)

print("najbolj potrošne funkcije glede na celoten čas izvajanja")
for i in range(0, 30):
	print(tottime_and_name[i])


print("\nnajbolj potrošne funkcije glede na posamezen čas izvajanja")
for i in range(0, 30):
	print(percall_and_name[i])


# import pstats
# p = pstats.Stats("edit_video.cprof")
# p.sort_stats('ncalls')
# p.print_stats(5)


