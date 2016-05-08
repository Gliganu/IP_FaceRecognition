from sklearn.datasets import fetch_lfw_people,fetch_olivetti_faces


def getLfwPeople(min_faces_per_person, resize):

    print('Loading LFW People')
    return fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)


def getOlivettiFaces():
    print('Loading Olivetti faces')
    return  fetch_olivetti_faces()
