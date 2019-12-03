import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import binom


def fast_non_dominated_sort(population_coords):
    '''
    Accepts a list of coordinates of solutions in score space,
    for each solution instance:

    Returns its rank according to the sorting result.

    Returns list of fronts, each of which is a list of
    solutions. fronts[0] is the nondominated front.
    '''

    fronts = []
    fronts.append([])

    popsize = population_coords.shape[0]
    ranks = np.zeros((popsize,))
    domination_table = np.zeros((popsize, popsize), dtype='bool')
    dominated_cardinal = np.zeros((popsize,))

    '''
    For each solution, modify 'domination_table' where
    True at row i column j means that solution i dominates j.
    Assign a number of solutions that dominate solution i
    to 'dominated_cardinal[i]' field.
    '''
    for i, p in enumerate(population_coords):
        for j, q in enumerate(population_coords):
            domination = dominates(p,q)
            if domination == 1:
                domination_table[i, j] = True
            elif domination == -1:
                dominated_cardinal[i] += 1
        if dominated_cardinal[i] == 0:
            ranks[i] = 1
            fronts[0].append(i)

    '''
    Peel fronts one by one, from nondominated to
    the worst one.
    '''
    i = 0
    while len(fronts[i]) != 0:
        next_front = []
        for pi in fronts[i]:
            p = population_coords[pi]
            for qi in np.argwhere(domination_table[pi,:]==True).T[0]:
                q = population_coords[qi]
                dominated_cardinal[qi] -= 1
                if dominated_cardinal[qi] == 0:
                    ranks[qi] = i + 2
                    next_front.append(qi)
        i += 1
        fronts.append(next_front)

    return fronts, ranks


def dominates(p,q):
    '''
    Returns 1 if p dominates q, -1
    if q dominates p, 0 otherwise.
    '''
    if p[0] >= q[0] and p[1] >= q[1]:
        return 1
    elif p[0] <= q[0] and p[1] <= q[1]:
        return -1
    else:
        return 0


def get_reference_coords(div, dim):
    '''
    Generates a list of coordinates that divide each
    goal axis by specified number of divisions.
    '''

    start_vector = np.ones((div+1,))
    start_vector[:div] = np.arange(0,1,1/div)
    coords = np.array([(i,j) for i,j in zip(start_vector, reversed(start_vector))])

    for i in range(3,dim+1):
        coords_update = np.zeros((int(binom(i + div - 1, div)),i))
        reduction_vector = np.zeros((coords.shape[1],))
        reduction_vector[-1] = 1
        offset = 0
        for j in range(div + 1):
            reduced_coords = coords - reduction_vector * j * 1/div
            erase = np.argwhere(reduced_coords < -0.0000001)[:,0]
            reduced_coords = np.delete(reduced_coords, erase, axis=0)
            h, w = reduced_coords.shape
            coords_update[offset:offset+h,:i-1] = reduced_coords
            coords_update[offset:offset+h,i-1] = j*1/div
            offset += h
        coords = coords_update

    return coords


def normalize(candidate_scores, nondom_scores):
    '''
    Normalizes scores of candidate solutions according
    to NSGA-III specification.
    '''

    '''
    This fragment is almost entirely copied from
    official implementation by Deb and his students
    since the paper itself is not very clear on this calculation.

    It basically finds solutions that are closest to each axis
    among nondominated solutions. Resulting points are saved as
    'maximum' points.
    '''

    '''
    TODO: solve singular matrix
    '''

    ideal_point = np.min(candidate_scores, axis=0)
    weights = np.eye(candidate_scores.shape[1])
    weights[weights==0] = 1e6

    asf = np.max(nondom_scores * weights[:,None,:], axis=2)
    I = np.argmin(asf, axis=1)

    maximum = nondom_scores[I,:]

    '''
    Find intercepts of a hyper plane generated by 'maximum'
    points.
    '''
    points = maximum - ideal_point
    b = np.ones(points.shape[1])
    try:
        plane = np.linalg.solve(points, b)
        intercepts = 1/plane
    except np.linalg.LinAlgError:
        index = np.random.randint(len(points))
        coords = points[index]
        intercepts = np.ones(len(coords)) * np.sum(coords)
        print('Linalg fallback')

    print('Intercepts', intercepts)

    return (candidate_scores - ideal_point)/intercepts, ideal_point, intercepts

def associate(reference_points, normalized_candidate_scores, passing_number):
    '''
    Associates each candidate with a closest line generated
    by a reference point.

    Outputs two arrays:
    'assoc_table' has the same number of rows as normalized_candidate_scores,
        its first column contains reference point index and second - 
        distance to the line generated by it.
    'ref_count' is a 1D array of same length as reference_points.
        For each ref point it contains the number of associated
        solutions from candidates EXCLUDING last front.
    '''

    dist_matrix = np.zeros((normalized_candidate_scores.shape[0], reference_points.shape[0]))
    ref_count = np.zeros(reference_points.shape[0])

    '''
    Calculate distance from each solution to each line.
    TODO: Reimplement in C or Go
    '''
    for i, c in enumerate(normalized_candidate_scores):
        for j, p in enumerate(reference_points):
            diff_vector = c - p.dot(c) * p/np.sum(p**2)
            dist_matrix[i,j] = np.sqrt(np.sum(diff_vector**2))

    sol_to_ref_assoc_table = np.zeros((normalized_candidate_scores.shape[0], 2))
    sol_to_ref_assoc_table[:,0] = np.argmin(dist_matrix, axis=1)
    sol_to_ref_assoc_table[:,1] = np.min(dist_matrix, axis=1)

    ref_to_sol_assoc_table = [[] for i in range(reference_points.shape[0])]
    ref_to_last_assoc_table = [[] for i in range(reference_points.shape[0])]

    for i, sol_to_ref in enumerate(sol_to_ref_assoc_table):
        r = int(sol_to_ref[0])
        ref_to_sol_assoc_table[ r ].append([i, sol_to_ref[1]])
        ref_to_sol_assoc_table[ r ].sort(key=lambda x:x[1])
        if i >= passing_number:
            ref_to_last_assoc_table[ r ].append([i, sol_to_ref[1]])
            ref_to_last_assoc_table[ r ].sort(key=lambda x:x[1])


    for i in range(passing_number):
        ref_count[np.int(sol_to_ref_assoc_table[i,0])] += 1

    return ref_to_sol_assoc_table, sol_to_ref_assoc_table, ref_to_last_assoc_table, ref_count

def niche(ref_count,
          ref_to_sol_assoc_table,
          ref_to_last_assoc_table,
          points_to_choose_number,
          candidates,
          passing_number):
    '''
    From the last front choose points which will be added
    to the next generation.
    TODO: clean up the code and fix bug with duplicates
    TODO: fix argwhere bug
    '''

    '''
    Inputs:

    ref_count - array of references to refpoints
        index - ref point index
        value - number of references to this ref point
    assoc_table - array of associations between refpoints and solutions
        index - index of a candidate
        first column - index of associated refpoint
        second colum - sitance to closest refpoint
    points_to_choose_number - number of points we have to choose
    candidates - indices of candidates
    passing_number - number of candidates that already pass
    '''
    new_population = candidates[:passing_number]
    last_front = candidates[passing_number:]

    k = 1
    while k <= points_to_choose_number:
        best_ref_points = np.argwhere(ref_count==np.min(ref_count)).T[0]

        best_ref_point = np.random.choice(best_ref_points)
        #last_front_best = np.argwhere(last_front_assoc_table[:,0]==best_ref_point).T[0] + passing_number
        #all_best = 
        last_front_best = [s[0] for s in ref_to_last_assoc_table[best_ref_point]]
        if len(last_front_best) != 0:
            if ref_count[best_ref_point] == 0:
                last_front_bestest = last_front_best[0]
                ref_to_last_assoc_table[best_ref_point].pop(0)
                new_population += [candidates[int(last_front_bestest)]]
            else:
                index = np.random.randint(len(last_front_best))
                last_front_bestest = last_front_best[index]
                ref_to_last_assoc_table[best_ref_point].pop(index)
                new_population += [candidates[int(last_front_bestest)]]
            ref_count[best_ref_point] += 1
            k+=1
        else:
            ref_count[best_ref_point] = 1e9

    return new_population


def visualize_ranks(population, figname, ranks=None):
    for i, s in enumerate(population):
        color = 'm'
        if type(ranks) is np.ndarray:
            if ranks[i] % 3 == 1:
                color = 'r'
            elif ranks[i] % 3 == 2:
                color = 'g'
            else:
                color = 'b'
        plt.scatter(*s, c=color)
    plt.savefig(figname)

def nsga3(initial_coords, div):

    popsize = initial_coords.shape[0]
    dim = initial_coords.shape[1]

    '''
    Perform nondominated sort
    '''
    fronts, ranks = fast_non_dominated_sort(initial_coords)

    '''
    Make a set of candidates that will compete for the
    next generation
    '''
    candidates = []
    cutoff_number = popsize // 2
    candidates_number = 0

    for f in fronts:
        if candidates_number + len(f) < cutoff_number:
            candidates = candidates + f
            candidates_number += len(f)
        elif candidates_number + len(f) == cutoff_number:
            return candidates + f
        else:
            passing_number = len(candidates)
            points_to_choose_number = cutoff_number - passing_number
            candidates = candidates + f
            candidates_number += len(f)
            last_front = f
            break
    '''
    'candidates_number' - the amount of solutions that compete for
    next generation
    'candidates' - these solutions len(candidates) = candidates_number
    '''


    '''
    Generate reference points
    '''
    reference_points = get_reference_coords(div, dim)

    '''
    Normalize candidate scores
    '''
    candidate_scores = initial_coords[candidates]
    nondom_scores = initial_coords[fronts[0]]

    normalized_candidate_scores,ideal_point,intercepts = normalize(candidate_scores, nondom_scores)

    '''
    For each candidate point find out its nearest reference line and
    distance to it
    '''
    ref_to_sol_assoc_table, assoc_table, ref_to_last_assoc_table, ref_count = associate(reference_points,
                                       normalized_candidate_scores,
                                       passing_number)

    '''
    Niching: delete extra points
    '''
    new_population = niche(ref_count,
                           ref_to_sol_assoc_table,
                           ref_to_last_assoc_table,
                           points_to_choose_number,
                           candidates,
                           passing_number)

    return new_population, fronts[0]


def main():
    init = np.random.random_sample((400, 5)) * (-1)
    start = time.time()
    solution = nsga3(init, 2)
    stop = time.time()
    print(stop-start, 's')


if __name__ == '__main__':
    main()


