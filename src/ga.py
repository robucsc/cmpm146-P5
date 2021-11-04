import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles

def convert_to_tuple(list):
    return(*list, )

class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.

        # difficulty curve
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )

        # for y in
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    def calculate_fitness_two(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.

        # difficulty curve
        coefficients = dict(
            meaningfulJumpVariance=0.3,
            negativeSpace=0.3,
            pathPercentage=0.3,
            emptyPercentage=0.3,
            linearity=-0.4,
            solvability=1.9
        )

        # for y in
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self


    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        # this is the heuristic
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        #
        e_max = 1
        e_count = 0
        e_ceil = 12
        pipe_ceil = 12
        pipe_count = 5
        plat_length = 3
        block_ceil = 9
        left_clean = 0
        right_clean = 10
        back_clean_left = width - 10
        back_clean_right = width
        left = 10
        right = width - 10
        plat_list = ['B', 'M', '?']
        mutate_rate = 3000

        # First do some cleaning to remove objects within space we want clear
        for y in range(height - 1):
            for x in range(left_clean, right_clean):
                genome[y][x] = "-"

        for y in range(height - 1):
            for x in range(back_clean_left, back_clean_right):
                genome[y][x] = "-"

        for y in range(height - 10):
            for x in range(left, right):
                genome[y][x] = "-"

        # Remove any unwanted options
        for y in range(height - 1):
            for x in range(left, right):
                if random.randint(1, mutate_rate) < 2:
                    genome[y][x] = random.choice(options)
                if genome[y][x] == "X" and y < height - 2:
                    genome[y][x] = "-"
                elif genome[y][x] == "E" and (y < e_ceil or e_count >= e_max):
                    genome[y][x] = "-"
                    if e_count >= e_max:
                        e_count += 1
                elif genome[y][x] in plat_list and (genome[y - 1][x] in plat_list or y > block_ceil):
                    genome[y][x] = "-"
                elif genome[y][x] == "|":
                    genome[y][x] = "-"
                elif genome[y][x] == "T" and y < pipe_ceil:
                    genome[y][x] = "-"

        # Add in wanted items
        for y in range(height - 1):
            for x in range(left, right):
                if genome[y][x] == "T":
                    y2 = y + 1
                    while y2 < height:
                        genome[y2][x] = "|"
                        y2 += 1
                elif genome[y][x] in plat_list:
                    x2 = x + 1
                    while x2 < plat_length:
                        genome[y][x2] = random.choice(plat_list)
                        x2 += 1

        """
        for y in range(height):
            for x in range(left, right):
                if (random.randint(1, 1000) < 2 ):
                    genome[y][x] = random.choice(options)

                if (genome[y][x] == "X" and y < height - 2):
                    genome[y][x] = "-"

                if (genome[y][x] == "E" and y < e_ceil):
                    genome[y][x] = "-"
                if (genome[y][x] == "E" and e_count >= e_max):
                    genome[y][x] = "-"
                    e_count += 1
                if (genome[y][x] == "M" or 'B' or '?'):
                    x2 = x + 1
                    while x2 < plat_length:
                        genome[y][x2] = random.choice(plat_list)
                        x2 += 1

                if (genome[y][x] ==  ("M" or 'B' or '?') and y > block_ceil ):
                    genome[y][x] = "-"

                if (genome[y][x] ==  ("M" or 'B' or '?') and  genome[y-1][x] == ("M" or 'B' or '?')):
                    genome[y][x] = "-"

                if (genome[y][x] == "|" and y < pipe_ceil):
                    genome[y][x] = "-"
                
                if (genome[y][x] == "|"):
                    genome[y][x] = "-"

                if (genome[y][x] == "T" and y < pipe_ceil ):
                    genome[y][x] = "-"
                
                if genome[y][x] == "T" and y < pipe_ceil:
                    y2 = y + 1
                    while y2 < height:
                        genome[y2][x] = "|"
                        y2 += 1
                if (y == height - 1):
                    genome[y][x] = "X"
                
        for y in range(height - 1):
            for x in range(left_clean, right_clean):
                genome[y][x] = "-"

        for y in range(height - 10):
            for x in range(left, right):
                genome[y][x] = "-"

        for y in range(height - 1):
            for x in range(back_clean_left, back_clean_right):
                genome[y][x] = "-"
        """

        genome[7][-7] = "v"
        genome[8:14][-5] = ["f"] * 6
        genome[14][0] = "m"
        genome[14:16][-1] = ["X", "X"]

        # print(genome)
        return genome

    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 5

        # cross over example
        # mid = int(len(self.genome) / 2)
        # strand_left = self.genome[:mid]
        # strand_right = other.genome[mid:]
        # new_genome = strand_left + strand_right

        # if (random.randint(1, 100) > 20):
        cross = random.randint(1, int(len(self.genome)))
        strand_left = self.genome[:cross]
        strand_right = other.genome[cross:]
        new_genome = strand_left + strand_right

        # for y in range(height):
        #     for x in range(left, right):
        #         if (random.randint(1, 100) > 20):
        #             new_genome[y][x] = other.genome[y][x]
        #         else:
        #             new_genome[y][x] = self.genome[y][x]

        child = self.mutate(new_genome)
        # child = new_genome

        # do mutation; note we're returning a one-element tuple here
        return (Individual_Grid(child),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-7] = "v"
        for col in range(8, 14):
            g[col][-7] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-7] = "v"
        g[8:14][-5] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)

def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

def tournament(population):
    nu_population = []
    random.shuffle(population)
    for i in range(0, len(population) - 1, 1):
        winner = population[i] if population[i].fitness() > population[i + 1].fitness() else population[i + 1]
        nu_population.append(winner)
    return nu_population

def tournament_loser(population):
    nu_population = []
    random.shuffle(population)
    for i in range(0, len(population) - 1, 1):
        loser = population[i] if population[i].fitness() < population[i + 1].fitness() else population[i + 1]
        nu_population.append(loser)
    return nu_population

def elite(population):
    elite_tune = 3.0
    nu_population = []
    for indivdual in population:
        if indivdual.fitness() > elite_tune:
            nu_population.append(indivdual)
    return nu_population

def two_pop(pop_one, pop_two):
    elite_population = []
    rabble_population = []
    elite_tune = 3.0

    full_pop = pop_one + pop_two

    cross = random.randint(1, int(len(pop_one)))
    pop_left = pop_one[:cross]
    if cross > int(len(pop_two)):
        cross = len(pop_two) - 1
    pop_right = pop_two[cross:]
    full_pop = pop_left + pop_right

    for indivdual in full_pop:
        if indivdual.fitness() >= elite_tune:
            elite_population.append(indivdual)
        else:
            rabble_population.append(indivdual)
    return (elite_population, rabble_population)

# def proportionate(population):
#
#     sum_of_fitness = sum(abs(p.fitness()) for p in population)
#     prev_probability = random.uniform(0, sum_of_fitness)
#     current_probability = 0
#     # print("sum: ", sum_of_fitness)
#     # Choose candidates from population fit for crossover
#     for p in population:
#         current_probability += abs(p.fitness())
#         # print("fitness: ", abs(p.fitness()))
#         if current_probability > prev_probability:
#             chosen.append(p)
#         # Crossover
#     for c in range(0, len(chosen) - 1, 1):
#         result = chosen[c].generate_children(chosen[c + 1])
#         results.append(result[0])
#
#     return results


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage= -0.7,
            linearity=-0.5,
            solvability=6.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        # we just made a second fitness function
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def calculate_fitness_two(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.

        # difficulty curve
        coefficients = dict(
            meaningfulJumpVariance=0.3,
            negativeSpace=0.3,
            pathPercentage=0.3,
            emptyPercentage=0.3,
            linearity=-0.4,
            solvability=1.9
        )

        # for y in
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    def fitness(self):
        # this is the heuristic
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.1 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                pass
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = [(random.randint(1, width - 2), "3_coin", random.randint(0, height - 1))]
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)

# change this to Individual_DE for the other func

# Individual = Individual_Grid
Individual = Individual_DE

def migration(pop_one, pop_two):
    cross = random.randint(1, int(len(pop_one.genome)))
    strand_left = pop_one.genome[:cross]
    strand_right = pop_two.genome[cross:]
    return strand_left + strand_right

def generate_successors(population):
    nu_population = []
    pop_size = 480
    population = tournament(population) + elite(population)
    print('population size ', len(population))
    if len(population) > pop_size:
        random.shuffle(population)
        population = population[:399]
    for i in range(0, len(population) - 1, 1):
        nu_population.append(population[i].generate_children(population[i + 1])[0])
    return nu_population

def call_gen_children(population):
    nu_population = []
    if len(population) > 400:
        random.shuffle(population)
        population = population[:399]
    for i in range(0, len(population) - 1, 1):
        nu_population.append(population[i].generate_children(population[i + 1])[0])
    return nu_population

def generate_successors_2pop(elite_population, rabble_population):
    elite_population = call_gen_children(tournament(elite_population))
    rabble_population = call_gen_children(tournament(rabble_population))
    return two_pop(elite_population, rabble_population)

def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization - can get to better results quicker
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition - creates a folder called levels and puts the gen levels in folder
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel -  a good place to mod
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population

def ga_2pop():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization - can get to better results quicker
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")

        cross = random.randint(1, int(len(population)))
        elite_population = population[:cross]
        rabble_population = population[cross:]

        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(elite_population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition - creates a folder called levels and puts the gen levels in folder
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population_elite, next_population_rabble = generate_successors_2pop(elite_population, rabble_population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel -  a good place to mod
                next_population_elite = pool.map(Individual.calculate_fitness, next_population_elite,
                                           batch_size)
                next_population_rabble = pool.map(Individual.calculate_fitness_two,
                                                 next_population_rabble,
                                                 batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                elite_population = next_population_elite
                rabble_population = next_population_rabble

        except KeyboardInterrupt:
            pass
    return (elite_population, rabble_population)

if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
