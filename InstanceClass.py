# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : Desktop
# File    : InstanceClass.py
# PATH    : sdp
# Author  : trisr
# Date    : 13/01/2023
# Description :
"""




"""
# Last commit ID   :
# Last commit date :
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #


# ******************************************************************************************************************** #
# Importations
import json
import numpy as np

# ******************************************************************************************************************** #
# Function definition
def init(data):
    def jobs_init(data):
        for job in data["jobs"]:
            job["due_date"] = job["due_date"] - 1

    def conges_init(data):
        for staff in data["staff"]:
            staff["vacations"] = list(np.array(staff["vacations"]) - 1)

    jobs_init(data)
    conges_init(data)


def extract_projets(data):
    jobs = data["jobs"]
    return jobs


def extract_collaborators(data):
    staff = data["staff"]
    return staff


def extract_qualifications(data):
    qualifications = data["qualifications"]
    return qualifications


def extract_gain(data):
    jobs = data["jobs"]
    gains = []
    for job in jobs:
        gains.append([int(job["name"][-1]), job["gain"]])
    gains = [y for x, y in sorted(gains)]
    return gains


def extract_penality(data):
    Penalties = []
    for jobs in data["jobs"]:
        row = np.zeros(data["horizon"])
        for i in range(data["horizon"]):
            if i > jobs["due_date"]:
                row[i] = (i - jobs["due_date"]) * jobs["daily_penalty"]
        Penalties.append(row)
    Penalties = np.array(Penalties)
    return Penalties


def extract_staff_qualifications(data):
    R_qualifications = []
    qualifications = extract_qualifications(data)
    for staff in data["staff"]:
        row = np.zeros(len(qualifications))
        for x in staff["qualifications"]:
            row[qualifications.index(x)] = 1
        R_qualifications.append(row)
    R_qualifications = np.array(R_qualifications)
    return R_qualifications


def extract_projets_qualifications(data):
    projets_qualifications = []
    qualifications = extract_qualifications(data)
    for job in data["jobs"]:
        row = np.zeros(len(qualifications))
        for x in job["working_days_per_qualification"]:
            row[qualifications.index(x)] = job["working_days_per_qualification"][x]
        projets_qualifications.append(row)
    projets_qualifications = np.array(projets_qualifications)
    return projets_qualifications


def extract_staff_conges(data):
    conges = []
    for staff in data["staff"]:
        row = np.zeros(data["horizon"])
        for x in staff["vacations"]:
            row[x] = 1
        conges.append(row)
    conges = np.array(conges)
    return conges


def assert_instance(instance):
    print("Testing if the script works on toy_instance")
    variables = instance.variables
    assert variables["NP"] == 5, "Problem with the number of project"
    assert variables["NC"] == 3, "Problem with the number of workers"
    assert variables["NA"] == 3, "Problem with the number of qualifications"
    assert variables["H"] == 5, "Problem with the horizon"
    assert (variables["GAIN"] == np.array([20, 15, 15, 20, 10])).all(), "Problem with the gain"
    assert (variables["PENALTIES"] == np.array([[0, 0, 0, 3, 6],
                                               [0, 0, 0, 3, 6],
                                               [0, 0, 0, 0, 3],
                                               [0, 0, 0, 3, 6],
                                               [0, 0, 0, 0, 0]])).all(), "Problem with the penalties"
    assert (variables["STAFF_QUALIFICATIONS"] == np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1]])).all(), "Problem with the gain"
    assert (variables["COST_PROJECT"] == np.array([[1, 1, 1], [1, 2, 0], [1, 0, 2],[0, 2, 1],[0, 0, 2]])).all(), "Problem with the cost of project"
    assert (variables["CONGES"] == np.array([[0, 0, 0, 0, 0],[1, 0, 0, 0, 0],[0, 1, 0,0,0]])).all(), "Problem with the gain"
    print("Test passed")

# ******************************************************************************************************************** #
# Class definition
class Instance:
    def __init__(self, path=None):
        self.path = path
        self.data = None

    def load_files(self, path=None):
        if path:
            file = open(path)
            self.data = json.load(file)
        else:
            assert self.path, "No path gived"
            file = open(self.path)
            self.data = json.load(file)

    def build_instance(self):
        if not (self.data):
            self.load_files()
        data = self.data
        init(data)
        self.variables = {}

        self.variables["NP"] = len(extract_projets(data))
        self.variables["NC"] = len(extract_collaborators(data))
        self.variables["NA"] = len(extract_qualifications(data))
        self.variables["H"] = data["horizon"]

        self.variables["GAIN"] = extract_gain(data)
        self.variables["PENALTIES"] = extract_penality(data)
        self.variables["STAFF_QUALIFICATIONS"] = extract_staff_qualifications(data)
        self.variables["COST_PROJECT"] = extract_projets_qualifications(data)
        self.variables["CONGES"] = extract_staff_conges(data)


# ******************************************************************************************************************** #
# Configuration


# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    instance = Instance(path="toy_instance.json")
    instance.build_instance()
    assert_instance(instance)
    print("Done")
