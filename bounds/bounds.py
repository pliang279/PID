import cvxpy as cp
import numpy as np
from scipy.special import rel_entr
from scipy.stats import entropy
from cvxpy import SCS

def solve_Q(P: np.ndarray):
    '''
    Compute optimal Q given 3d array P 
    with dimensions coressponding to x1, x2, and y respectively
    '''
    Py = P.sum(axis=0).sum(axis=0)
    Px1 = P.sum(axis=1).sum(axis=1)
    Px2 = P.sum(axis=0).sum(axis=1)
    Px2y = P.sum(axis=0)
    Px1y = P.sum(axis=1)
    Px1y_given_x2 = P/P.sum(axis=(0,2),keepdims=True)
 
    Q = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) for i in range(P.shape[2])]
    Q_x1x2 = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) for i in range(P.shape[2])]

    # Constraints that conditional distributions sum to 1
    sum_to_one_Q = cp.sum([cp.sum(q) for q in Q]) == 1

    # Brute force constraints # 
    # [A]: p(x1, y) == q(x1, y) 
    # [B]: p(x2, y) == q(x2, y)

    # Adding [A] constraints
    A_cstrs = []
    for x1 in range(P.shape[0]):
        for y in range(P.shape[2]):
            vars = []
            for x2 in range(P.shape[1]):
            vars.append(Q[y][x1, x2])
            A_cstrs.append(cp.sum(vars) == Px1y[x1,y])
  
    # Adding [B] constraints
    B_cstrs = []
    for x2 in range(P.shape[1]):
        for y in range(P.shape[2]):
            vars = []
            for x1 in range(P.shape[0]):
            vars.append(Q[y][x1, x2])
            B_cstrs.append(cp.sum(vars) == Px2y[x2,y])

    # KL divergence
    Q_pdt_dist_cstrs = [cp.sum(Q) / P.shape[2] == Q_x1x2[i] for i in range(P.shape[2])]


    # objective
    obj = cp.sum([cp.sum(cp.rel_entr(Q[i], Q_x1x2[i])) for i in range(P.shape[2])])
    # print(obj.shape)
    all_constrs = [sum_to_one_Q] + A_cstrs + B_cstrs + Q_pdt_dist_cstrs
    prob = cp.Problem(cp.Minimize(obj), all_constrs)
    try:
        prob.solve(verbose=False)
    except:
        prob.solve(solver=SCS, verbose=False)
    return np.stack([q.value for q in Q],axis=2)

def convert_data_to_distribution(x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
    assert x1.size == x2.size
    assert x1.size == y.size

    numel = x1.size
    
    x1_discrete, x1_raw_to_discrete = extract_categorical_from_data(x1.squeeze())
    x2_discrete, x2_raw_to_discrete = extract_categorical_from_data(x2.squeeze())
    y_discrete, y_raw_to_discrete = extract_categorical_from_data(y.squeeze())

    joint_distribution = np.zeros((len(x1_raw_to_discrete), len(x2_raw_to_discrete), len(y_raw_to_discrete)))

    for i in range(numel):
        joint_distribution[x1_discrete[i], x2_discrete[i], y_discrete[i]] += 1
    joint_distribution /= np.sum(joint_distribution)

    return joint_distribution, (x1_raw_to_discrete, x2_raw_to_discrete, y_raw_to_discrete)

def extract_categorical_from_data(x):
    supp = set(x)
    raw_to_discrete = dict()
    for i in supp:
        raw_to_discrete[i] = len(raw_to_discrete)
    discrete_data = [raw_to_discrete[x_] for x_ in x]

    return discrete_data, raw_to_discrete 

def MI(P: np.ndarray):
    ''' P has 2 dimensions '''
    margin_1 = P.sum(axis=1)
    margin_2 = P.sum(axis=0)
    outer = np.outer(margin_1, margin_2)
    return np.sum(rel_entr(P, outer))

def CoI(P:np.ndarray):
    ''' P has 3 dimensions, in order X1, X2, Y '''
    # MI(Y; X1)
    A = P.sum(axis=1)
    # MI(Y; X2)
    B = P.sum(axis=0)
    # MI(Y; (X1, X2))
    C = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))
    return MI(A) + MI(B) - MI(C)

def CI(P, Q):
    assert P.shape == Q.shape
    P_ = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))
    Q_ = Q.transpose([2, 0, 1]).reshape((Q.shape[2], Q.shape[0]*Q.shape[1]))
    return MI(P_) - MI(Q_)

def UI(P, cond_id=0):
    ''' P has 3 dimensions, in order X1, X2, Y 
    We condition on X1 if cond_id = 0, if 1, then X2.
    '''
    P_ = np.copy(P)
    sum = 0.

    if cond_id == 0:
        J= P.sum(axis=(1,2)) # marginal of x1
        for i in range(P.shape[0]):
            sum += MI(P[i,:,:]/P[i,:,:].sum()) * J[i]
    elif cond_id == 1:
        J= P.sum(axis=(0,2)) # marginal of x1
        for i in range(P.shape[1]):
            sum += MI(P[:,i,:]/P[:,i,:].sum()) * J[i]
    else:
        assert False

    return sum

def test(P):
    Q = solve_Q(P)
    redundancy = CoI(Q) / np.log(2)
    print('Redundancy:', redundancy)
    unique_1 = UI(Q, cond_id=1) / np.log(2)
    print('Unique1', unique_1)
    unique_2 = UI(Q, cond_id=0) / np.log(2)
    print('Unique2', unique_2)
    synergy = CI(P, Q) / np.log(2)
    print('Synergy', synergy)
    return redundancy, unique_1, unique_2, synergy

def get_quantities(P):
    all = {}
    all['Py'] = np.sum(np.sum(P, axis=0, keepdims=True), axis=1, keepdims=True)
    all['Px1x2'] = np.sum(P, axis=2, keepdims=True)
    all['Px1y'] = np.sum(P, axis=1, keepdims=True)
    all['Px2y'] = np.sum(P, axis=0, keepdims=True)
    all['Px1'] = np.sum(all['Px1y'], axis=2, keepdims=True)
    all['Px2'] = np.sum(all['Px2y'], axis=2, keepdims=True)
    all['Py_given_x1'] = all['Px1y']/all['Px1']
    all['Py_given_x2'] = all['Px2y']/all['Px2']
    all['Py_given_x1x2'] = P/all['Px1x2']
    all['Px1_given_y'] = all['Px1y']/all['Py']
    all['Px2_given_y'] = all['Px2y']/all['Py']
    all['Px1x2_given_y'] = P/all['Py']
    all['Px1_given_x2y'] = P/all['Px2y']
    all['Px2_given_x1y'] = P/all['Px1y']

    all['H_y'] = entropy(all['Py'].squeeze(), base=2)
    all['H_x1'] = entropy(all['Px1'].squeeze(), base=2)
    all['H_x2'] = entropy(all['Px2'].squeeze(), base=2)
    all['H_x1x2'] = entropy(all['Px1x2'].flatten(), base=2)

    all['I_x1x2'] = np.sum(rel_entr(all['Px1x2'].squeeze(axis=2), (all['Px1']*all['Px2']).squeeze(axis=2))) / np.log(2)
    all['I_x1y'] = np.sum(rel_entr(all['Px1y'].squeeze(axis=1), (all['Px1']*all['Py']).squeeze(axis=1))) / np.log(2)
    all['I_x2y'] = np.sum(rel_entr(all['Px2y'].squeeze(axis=0), (all['Px2']*all['Py']).squeeze(axis=0))) / np.log(2)

    H_y_given_x1x2 = entropy(all['Py_given_x1x2'], base=2, axis=2)
    all['H_y_given_x1x2'] = np.einsum('ij,ij', H_y_given_x1x2, all['Px1x2'].squeeze(axis=2))
    H_x1_given_x2y = entropy(all['Px1_given_x2y'], base=2, axis=0)
    all['H_x1_given_x2y'] = np.einsum('ij,ij', H_x1_given_x2y, all['Px2y'].squeeze(axis=0))
    H_x2_given_x1y = entropy(all['Px2_given_x1y'], base=2, axis=1)
    all['H_x2_given_x1y'] = np.einsum('ij,ij', H_x2_given_x1y, all['Px1y'].squeeze(axis=1))

    all['I_x1x2_given_y'] = all['H_x1'] - all['I_x1y'] - all['H_x1_given_x2y']
    all['I_x1x2y'] = all['I_x1x2'] - all['I_x1x2_given_y']
    all['I_x1y_given_x2'] = all['I_x1y'] - all['I_x1x2y']
    all['I_x2y_given_x1'] = all['I_x2y'] - all['I_x1x2y']

    r, u1, u2, s = test(P)
    all['R'] = r
    all['U2'] = u1
    all['U1'] = u2
    all['S'] = s
    return all

def majorization_bound(Px1x2_, Py_):
    Px1x2 = Px1x2_.copy().flatten()
    Px1x2.sort()
    Px1x2 = Px1x2[::-1]
    Py = Py_.copy()
    Py.sort()
    Py = Py[::-1]

    # Augment so they are the same size
    fsize = max([Px1x2.size, Py.size]) + 1
    Px1x2 = np.pad(Px1x2, (0, fsize-Px1x2.size))
    Py = np.pad(Py, (0, fsize-Py.size))
    cumX1X2 = np.cumsum(Px1x2)
    cumY = np.cumsum(Py)

    z_prev = min([Px1x2[0], Py[0]])
    z = [z_prev]
    for i in range(1, fsize):
        z_prev = min([cumX1X2[i], cumY[i]]) - sum(z)
        z.append(z_prev)
    return entropy(np.array(z), base=2)

def upper_bound(P, all_p):
    # bound1
    Px1x2 = np.sum(P, axis=2)
    Py = np.sum(np.sum(P, axis=0), axis=0)
    H_min1 = majorization_bound(Px1x2, Py) # Get 3d estimate

    # bound2
    Px1y = np.sum(P, axis=1)
    Px2 = np.sum(np.sum(P, axis=0), axis=-1)
    H_min2 = majorization_bound(Px1y, Px2) # Get 3d estimate

    # bound3
    Px2y = np.sum(P, axis=0)
    Px1 = np.sum(np.sum(P, axis=-1), axis=-1)
    H_min3 = majorization_bound(Px2y, Px1) # Get 3d estimate

    H_min = max([H_min1, H_min2, H_min3])
    H_x1x2 = entropy(Px1x2.flatten(), base=2)
    H_y = entropy(Py, base=2)
    max_MI = all_p['H_x1x2'] + all_p['H_y'] - H_min

    return max_MI - all_p['R'] - all_p['U1'] - all_p['U2']

def solve_Q_other(P):
  # P is (x1 x2 y), constraints (x1y) (x2y), objective H(y|x1x2)
  # P1 is (x2 y x1), constraints (x2x1)(yx1), objective H(x1|x2y)
  P1 = np.swapaxes(np.swapaxes(P,1,2), 0, 2)
  # P2 is (x1 y x2), constraints (x1x2)(yx2), objective H(x2|x1y)
  P2 = np.swapaxes(P,1,2)
  Q1 = solve_Q(P1) # Q1 is in (x2 y x1) space
  Q2 = solve_Q(P2) # Q2 is in (x1 y x2) space
  return Q1, Q2

def get_bounds(P):
    r, u1, u2, s = test(P)
    # print ('r=', r, 'u1=', u1, 'u2=', u2, 's=', s, 'total=', r+u1+u2+s)

    all_p = get_quantities(P)
    Py_given_x1 = all_p['Py_given_x1'].squeeze(axis=1)
    Py_given_x2 = all_p['Py_given_x2'].squeeze(axis=0)
    diff = np.zeros((Py_given_x1.shape[0], Py_given_x2.shape[0]))
    for i in range(Py_given_x1.shape[0]):
        for j in range(Py_given_x2.shape[0]):
            diff[i][j] = np.sum((Py_given_x1[i] - Py_given_x2[j]) ** 2)
    diff = np.einsum('ij,ij', all_p['Px1x2'].squeeze(axis=-1), diff)

    Q1, Q2 = solve_Q_other(P)
    all_q1 = get_quantities(Q1)
    all_q2 = get_quantities(Q2)

    # this is based on S = R - I(X1;X2) + I(X1;X2|Y), then min I(X1;X2|Y) = min H(X1) - I(X1;Y) - H(X1|X2,Y)
    I_x1x2_given_y_min_from1 = all_p['H_x1'] - all_p['I_x1y'] - all_q1['H_y_given_x1x2']
    I_x1x2_given_y_min_from2 = all_p['H_x2'] - all_p['I_x2y'] - all_q2['H_y_given_x1x2']
    I_x1x2_given_y = all_p['I_x1x2_given_y']
    I_x1x2_given_y_min = max(I_x1x2_given_y_min_from1, I_x1x2_given_y_min_from2)
    # print (I_x1x2_given_y_min_from1, I_x1x2_given_y)
    lower_R = all_p['R'] - all_p['I_x1x2'] + I_x1x2_given_y_min_from1

    # these are based on S = I(X1;Y|X2) - U1 and I(X2;Y|X1) - U2, then min I(X1;Y|X2) = min H(X1) - I(X1;X2) - H(X1|X2,Y)
    lower_U1 = all_p['H_x1'] - all_p['I_x1x2'] - all_q1['H_y_given_x1x2'] - all_p['U1']
    lower_U2 = all_p['H_x2'] - all_p['I_x1x2'] - all_q2['H_y_given_x1x2'] - all_p['U2']

    # this is disagreement - max(U1,U2)
    lower_diff = diff - max(all_p['U1'], all_p['U2'])

    # this is min-entropy coupling
    upper = upper_bound(P, all_p)

    all_bounds = {}
    all_bounds['lower_R'] = lower_R
    all_bounds['lower_U1'] = lower_U1
    all_bounds['lower_U2'] = lower_U2
    all_bounds['lower_diff'] = lower_diff
    all_bounds['upper'] = upper

    return all_p, all_bounds