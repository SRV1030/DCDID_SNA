# -*- coding: utf-8 -*-
"""
@author: 1912101 Aditya Soni, 1912104 Aditya Agarwal, 1912106 Abhishek Bharadwaj, 1912158 Sourabh Shah, 1912177 Ayesha Nashim
"""
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import defaultdict
import time
import datetime
from itertools import count
from sklearn import metrics
import math
import matplotlib.pyplot as plt  # for drawing


def str_to_int(x):
    return [[int(v) for v in line.split()] for line in x]


# The input community format is {node: community name}
def node_addition(G, addnodes, communities):
    change_comm = set()  # Store community tags that the structure may find changed
    # processed edges, need to remove processed edges from added edges
    processed_edges = set()

    for u in addnodes:
        neighbors_u = G.neighbors(u)
        neig_comm = set()  # Neighbor's community label
        pc = set()
        for v in neighbors_u:
            neig_comm.add(communities[v])
            pc.add((u, v))
            # There is one edge in the undirected graph, and it is convenient to add two times
            pc.add((v, u))
        if len(neig_comm) > 1:  # Indicates that this joining node is not within the community
            change_comm = change_comm | neig_comm
            lab = max(communities.values())+1
            communities.setdefault(u, lab)  # assign a community label to v
            change_comm.add(lab)
        else:
            # Indicates that the node is inside the community, or only connected to one community
            if len(neig_comm) == 1:
                # Add the node to this community
                communities.setdefault(v, neig_comm[0])
                processed_edges = processed_edges | pc
            else:
                # The newly added node is not connected to other nodes, assign a new community label
                communities.setdefault(v, max(communities.values())+1)

    # Returns possibly changed communities, processed edges and latest community structure.
    return change_comm, processed_edges, communities


def node_deletion(G, delnodes, communities):  # tested, correct
    change_comm = set()  # Store community tags that the structure may find changed
    # processed edges, need to remove processed edges from added edges
    processed_edges = set()
    for u in delnodes:
        neighbors_u = G.neighbors(u)
        neig_comm = set()  # Neighbor's community label
        for v in neighbors_u:
            neig_comm.add(communities[v])
            processed_edges.add((u, v))
            processed_edges.add((v, u))
        del communities[u]  # delete nodes and communities
        change_comm = change_comm | neig_comm
    # Returns possibly changed communities, processed edges and latest community structure.
    return change_comm, processed_edges, communities


# If the added edge is inside the community and will not cause community changes, it will not be processed, otherwise it will be marked
def edge_addition(addedges, communities):
    change_comm = set()  # Store community tags that the structure may find changed
# print addedges
# print communities
    for item in addedges:
        neig_comm = set()  # Neighbor's community label
        # Determine the community where the nodes at both ends of one side are located
        neig_comm.add(communities[item[0]])
        neig_comm.add(communities[item[1]])
        if len(neig_comm) > 1:  # Indicates that this joining edge is not within the community
            change_comm = change_comm | neig_comm
    return change_comm  # Returns the community that may have changed,


# If deleting an edge may cause community changes within the community, it will not change outside the community
def edge_deletion(deledges, communities):
    change_comm = set()  # Store community tags that the structure may find changed
    for item in deledges:
        neig_comm = set()  # Neighbor's community label
        # Determine the community where the nodes at both ends of one side are located
        neig_comm.add(communities[item[0]])
        neig_comm.add(communities[item[1]])
        if len(neig_comm) == 1:  # Indicates that this joining edge is not within the community
            change_comm = change_comm | neig_comm
    return change_comm  # Returns the community that may have changed


def getchangegraph(all_change_comm, newcomm, Gt):
    Gte = nx.Graph()
    com_key = newcomm.keys()
    for v in Gt.nodes():
        if v not in com_key or newcomm[v] in all_change_comm:
            Gte.add_node(v)
            neig_v = Gt.neighbors(v)
            for u in neig_v:
                if u not in com_key or newcomm[u] in all_change_comm:
                    Gte.add_edge(v, u)
                    Gte.add_node(u)

    return Gte


def CDID(Gsub, maxlabel):  # G_sub is a subgraph, run information dynamics on subgraphs that may change the structure, maxlabel is the maximum label that does not change the community structure

    # initial information
    Neigb = {}
    info = 0
    # average degree, maximum degree
    # avg_d = 0
    max_deg = 0
    N = Gsub.number_of_nodes()
    deg =[] 
    deg2=Gsub.degree()
    for d in  Gsub.degree():
        deg.append(d[1])

    max_deg = max(deg)
    print("//////degrees")
    print(max_deg)
    print(deg)
    # avg_d = sum(deg) * 1.0 / N

    ti = 1
    list_I = {}  # Store the information of each node, the initial is the degree of each node, and each iteration continues to change dynamically
    maxinfo = 0
    starttime = datetime.datetime.now()
    for v in Gsub.nodes():
        if deg2[v] == max_deg:
            info_t = 1 + ti * 0
            ti = ti + 1
# print v,max_deg,info_t
            maxinfo = info_t
        else:
            info_t = deg2[v] * 1.0 / max_deg
            # info_t=round(random.uniform(0,1),3)
        # info_t=deg[v]*1.0/max_deg
        list_I.setdefault(v, info_t)
        Neigb.setdefault(v, Gsub.neighbors(v))  # Neighbors of node v
        info += info_t
    node_order = sorted(list_I.items(), key=lambda t: t[1], reverse=True)
    node_order_list = list(zip(*node_order))[0]
    # Calculate the similarity between nodes, the Jaccard coefficient

    def sim_jkd(u, v):
        list_v = Gsub.neighbors(v)
        list_v.append(v)
        list_u = Gsub.neighbors(u)
        list_u.append(u)
        t = set(list_v)
        s = set(list_u)

        return len(s & t) * 1.0 / len(s | t)
    # Calculate the number of hop2 between nodes

    def hop2(u, v):
        list_v = (Neigb[v])
        list_u = (Neigb[u])
        t = set(list_v)
        s = set(list_u)
        return len(s & t)

    st = {}  # store the similarity
    hops = {}  # store hop2 number

    hop2v = {}  # Store the ratio of hop2 numbers
    sum_s = {}  # Store the sum of the neighbor similarity of each node
    avg_sn = {}  # Store the local average similarity of each node, local refers to the neighbor nodes
    avg_dn = {}  # Store the local average degree of each node

    for v, Iv in list_I.items():
        sum_v = 0
        sum_deg = 0
        tri = nx.triangles(Gsub, v) * 1.0
        listv = Neigb[v]
        num_v = len(list(listv))
        sum_deg += deg2[v]

        for u in listv:
            keys = str(v) + '_' + str(u)
            p = st.setdefault(keys, sim_jkd(v, u))
            h2 = hop2(v, u)
            hops.setdefault(keys, h2)
            if tri == 0:
                if deg2[v] == 1:
                    hop2v.setdefault(keys, 1)
                else:
                    hop2v.setdefault(keys, 0)
            else:
                hop2v.setdefault(keys, h2/tri)

            sum_v += p
            sum_deg += deg2[u]

        sum_s.setdefault(v, sum_v)
        avg_sn.setdefault(v, sum_v * 1.0 / num_v)
        avg_dn.setdefault(v, sum_deg * 1.0 / (num_v + 1))
    # print('begin loop')

    # oldinfo = 0
    info = 0
    t = 0
    while 1:
        info = 0
        t = t + 1
        Imax = 0

        for i in range(len(node_order_list)):
            v = node_order_list[i]
            Iv = list_I[v]
            for u in Neigb[v]:
                # p=sim_jkd(v,u)
                keys = str(v) + '_' + str(u)

                Iu = list_I[u]
                if Iu - Iv < 0:
                    # It=It*1.0/E
                    It = 0
                else:
                    It = (math.exp(Iu - Iv) - 1)
                # It=It*1.0*deg[u]/(deg[v]+deg[u])
                if It < 0.0001:
                    It = 0
                fuv = It
                # print(fuv)
                p = st[keys]
                p1 = p * hop2v[keys]
                Iin = p1 * fuv
                Icost = avg_sn[v] * fuv * (1 - p) / avg_dn[v]
                # Icost=avg_s*fuv*avg_c/avg_d
                # Icost=(avg_sn[v])*fuv/avg_dn[v]

                Iin = Iin - Icost
                if Iin < 0:
                    Iin = 0
                Iv = Iv + Iin
                # print(v,u,Iin,Icost,Iv,Iu,It)
                if Iin > Imax:
                    Imax = Iin

            if Iv > maxinfo:
                Iv = maxinfo
            list_I[v] = Iv
            # print(v,u,Iin,Iv,Iu,tempu[0],pu,tempu[1],fuv)
            info += list_I[v]
        # if v==3:
        # print(v,Iv)

        if Imax < 0.0001:
            break

    endtime = datetime.datetime.now()
    # print ('time:', (endtime - starttime).seconds)
    # Group division ************************************************ ****************

    queue = []
    order = []
    community = {}
    lab = maxlabel
    number = 0
    for v, Info in list_I.items():
        if v not in community.keys():
            lab = lab + 1
            queue.append(v)
            order.append(v)
            community.setdefault(v, lab)
            number = number + 1
            while len(queue) > 0:
                node = queue.pop(0)
                for n1 in Neigb[node]:
                    if (not n1 in community.keys()) and (not n1 in queue):
                        if abs(list_I[n1] - list_I[node]) < 0.001:
                            queue.append(n1)
                            order.append(n1)
                            community.setdefault(n1, lab)
                            number = number + 1
        if number == N:
            break

            # print (order)
            # print(community)
    order_value = [community[k] for k in sorted(community.keys())]
    commu_num = len(set(order_value))  # number of communities
    endtime1 = datetime.datetime.now()
    print('Social division ends')
    print(list_I)
    # print('community number:', commu_num)
    print('alltime:', (endtime1 - starttime).seconds)
    return community


# Convert community format to, label as primary key, node as value
def conver_comm_to_lab(comm1):
    overl_community = {}
    for node_v, com_lab in comm1.items():
        if com_lab in overl_community.keys():
            overl_community[com_lab].append(node_v)
        else:
            overl_community.update({com_lab: [node_v]})
    return overl_community


def getscore(comm_va, comm_list):
    actual = []
    baseline = []
    # groundtruth, j represents each community, j is the community name
    for j in range(len(comm_va)):
        for c in comm_va[j]:  # Each node in the community, representing each node
            flag = False
            # The detected community, k is the community name
            for k in range(len(comm_list)):
                if c in comm_list[k] and flag == False:
                    flag = True
                    actual.append(j)
                    baseline.append(k)
                    break
    print('nmi', metrics.normalized_mutual_info_score(actual, baseline))
    print('ari', metrics.adjusted_rand_score(actual, baseline))


def drawcommunity(g, partition, filepath):
    pos = nx.spring_layout(g)
    count1 = 0
    t = 0

    for com in set(partition.values()):
        count1 = count1 + 1.
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        print(list_nodes)
        nx.draw_networkx_nodes(g, pos, list_nodes, node_size=220)
        nx.draw_networkx_labels(g, pos)
        t = t+1
    
    labels = nx.get_edge_attributes(g,'weight')
    nx.draw_networkx_labels(g,pos)
    nx.draw_networkx_edges(g, pos,alpha=0.5)
    plt.savefig(filepath)
    plt.show()


################################################## ##########
# ------------main-----------------
edges_added = set()
edges_removed = set()
nodes_added = set()
nodes_removed = set()
G = nx.Graph()

# Edge Path
edge_file = '15node_t04.txt'
# Path to the directory
path = 'DCDID1/data/test1/'
# Adding nodes to the graph with edges
with open(path+edge_file, 'r') as f:

    edge_list = f.readlines()
    for edge in edge_list:
        edge = edge.split()
        G.add_node(int(edge[0]))
        G.add_node(int(edge[1]))
        G.add_edge(int(edge[0]), int(edge[1]))
G = G.to_undirected()

# initial graph

print('Time Slice NetworkG0******************************************** *')
nx.draw_networkx(G)
fpath = 'DCDID1/data/pic/G_0.png'
plt.savefig(fpath)
# Output method 1: save the image as a png format image file
plt.show()
# print G.edges()
# comm_file='switch.t01.comm'
comm_file = '15node_comm_t04.txt'
with open(path+comm_file, 'r') as f:
    comm_list = f.readlines()
    comm_list = str_to_int(comm_list)
comm = {}  # Used to store the detected community structure in the format {node: community label}
comm = CDID(G, 0)  # initial community
# drawing community
print('Community C0 of T0 time slice******************************************** ****')
print(comm)
drawcommunity(G, comm, 'DCDID1/data/pic/community_0.png')
initcomm = conver_comm_to_lab(comm)
comm_va = list(initcomm.values())
getscore(comm_va, comm_list)
start = time.time()
G1 = nx.Graph()
G2 = nx.Graph()
G1 = G
# filename='switch.t0'
filename = '15node_'
for i in range(2, 5):
    print('begin loop:', i-1)
    # comm_new_file=open(path+'output_new_'+str(i)+'.txt','r')
# comm_new_file=open(path+filename+str(i)+'.comm','r')
    comm_new_file = open(path+filename+'comm_t0'+str(i)+'.txt', 'r')
    if i < 10:
        # edge_list_old_file=open(path+'switch.t0'+str(i-1)+'.edges','r')
        # edge_list_old=edge_list_old_file.readlines()
        # edge_list_new_file=open(path+filename+str(i)+'.edges','r')
        edge_list_new_file = open(path+filename+'t0'+str(i)+'.txt', 'r')
        edge_list_new = edge_list_new_file.readlines()
        comm_new = comm_new_file.readlines()
    elif i == 10:
        # edge_list_old_file=open(path+'switch.t09.edges','r')
        # edge_list_old=edge_list_old_file.readlines()
        edge_list_new_file = open(path+'switch.t10.edges', 'r')
        edge_list_new = edge_list_new_file.readlines()
        comm_new = comm_new_file.readlines()
    else:
        # edge_list_old_file=open(path+'switch.t'+str(i-1)+'.edges','r')
        # edge_list_old=edge_list_old_file.readlines()
        edge_list_new_file = open(path+'switch.t'+str(i)+'.edges', 'r')
        edge_list_new = edge_list_new_file.readlines()
        comm_new = comm_new_file.readlines()
    comm_new = str_to_int(comm_new)

# for line in edge_list_old:
# temp = line.strip().split()
#
# G1.add_edge(int(temp[0]),int(temp[1]))
    for line in edge_list_new:
        temp = line.strip().split()
        G2.add_edge(int(temp[0]), int(temp[1]))
    print('T'+str(i-1)+'time slice network G'+str(i-1) + '********************** ************************')
    nx.draw_networkx(G2)
    fpath = 'DCDID1/data/pic/G_' + \
        str(i-1)+'.png'
    # Output method 1: save the image as a png format image file
    plt.savefig(fpath)
    plt.show()
# total_nodes = previous_nodes.union(current_nodes)#The total number of nodes in the current time slice and the previous time slice, the two sets are related
    total_nodes = set(G1.nodes()) | set(G2.nodes())
# current_nodes.add(1002)
# previous_nodes.add(1001)

    nodes_added = set(G2.nodes())-set(G1.nodes())
    print('Add node set to: ', nodes_added)
    nodes_removed = set(G1.nodes())-set(G2.nodes())
    print('Remove node set as:', nodes_removed)
# print ('G2', G2.nodes())
# print ('G1', G1.nodes())
# print ('add node',nodes_added)
# print ('remove node',nodes_removed)
    edges_added = set(G2.edges())-set(G1.edges())
    print('Added edge set is: ', edges_added)
    edges_removed = set(G1.edges())-set(G2.edges())
    print('Delete edge set: ', edges_removed)
# print ('add edges',edges_added)
# print ('remove edges',edges_removed)
# print len(G1.edges())
# print len(edges_added), len(edges_removed)
    all_change_comm = set()
    #Add node processing ############################################## ################
    addn_ch_comm, addn_pro_edges, addn_commu = node_addition(G2, nodes_added, comm)
# print ('addnode_community',addn_commu)
# print edges_added
# print addn_pro_edges
    edges_added = edges_added-addn_pro_edges # remove processed edges
# print edges_added
    all_change_comm = all_change_comm | addn_ch_comm
# print('addn_ch_comm',addn_ch_comm)

    #Delete node processing ############################################## ################
# print('nodes_removed',nodes_removed)
    deln_ch_comm, deln_pro_edges, deln_commu = node_deletion(
        G1, nodes_removed, addn_commu)
    all_change_comm = all_change_comm | deln_ch_comm
    edges_removed = edges_removed-deln_pro_edges
# print('deln_ch_comm',deln_ch_comm)
# print ('delnode_community',deln_commu)
    #Add edge processing ############################################## ###############
# print('edges_added',edges_added)
    adde_ch_comm = edge_addition(edges_added, deln_commu)
    all_change_comm = all_change_comm | adde_ch_comm
# print('all_change_comm',all_change_comm)
    #delete edge processing ############################################## ###############
    dele_ch_comm = edge_deletion(edges_removed, deln_commu)
    all_change_comm = all_change_comm | dele_ch_comm
# print('all_change_comm',all_change_comm)
    unchangecomm = () # Unchanged community tag
    newcomm = {} # The format is {node:community}
    newcomm = deln_commu # Add edges and delete edges, just process on existing nodes, no new nodes will be added, nodes will be deleted (previously processed)
    unchangecomm = set(newcomm.values())-all_change_comm
    unchcommunity = {key: value for key, value in newcomm.items(
    ) if value in unchangecomm} # unchanged community : tags and nodes
    # Find the subgraph corresponding to the changed community, then use information dynamics on the subgraph to find the new community structure, add the unchanged community structure, and get the new community structure.
# print('change community:',all_change_comm)
    Gtemp = nx.Graph()
    Gtemp = getchangegraph(all_change_comm, newcomm, G2)
    unchagecom_maxlabe = 0
    if len(unchangecomm) > 0:
        unchagecom_maxlabe = max(unchangecomm)
# print('subG', Gtemp.edges())
    if Gtemp.number_of_edges() < 1: # community has not changed
        comm = newcomm
    else:
        getnewcomm = CDID(Gtemp, unchagecom_maxlabe)
        print('T'+str(i-1)+'time slice delta_g'+str(i-1) +
              '************************************************')
        nx.draw_networkx(Gtemp)
        fpath = 'DCDID1/data/pic/delta_g' + \
            str(i-1)+'.png'
        plt.savefig(fpath)
        plt.show()

# print('newcomm', getnewcomm)
        # Merge community structure, unchanged plus newly acquired
# mergecomm=dict(unchcommunity, **getnewcomm )#The format is {node:community}
        d = dict(unchcommunity)
        d.update(getnewcomm)
        comm = dict(d) # Take the currently obtained community structure as the next community input
        print('T'+str(i-1)+'time slice network community structure C'+str(i-1) +
              '************************************************')
        drawcommunity(G2, comm, 'DCDID1/data/pic/community_'+str(i-1)+'.png')
# print ('getcommunity:',conver_comm_to_lab(comm))
    getscore(list(conver_comm_to_lab(comm).values()), comm_new)
    print('community number:', len(set(comm.values())))
    print(comm)
    G1.clear()
    G1.add_edges_from(G2.edges())
    G2.clear()
print('all done')