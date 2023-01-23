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
import os

# ******************************************************************************************************************** #
# Function definition


def jobs_init(data):
    """
    On enlève 1 aux dates de rendu des projets pour tenir compte que l'on commence à travailler au jour 1.
    """
    for job in data["jobs"]:
        job["due_date"] = job["due_date"] - 1


def conges_init(data):
    """
    On met en forme les vacances des employés pour tenir compte que l'on commence à travailler le jour 1.
    """
    for staff in data["staff"]:
        staff["vacations"] = list(np.array(staff["vacations"]) - 1)


def init(data):
    """
    Applique des pré-traitements aux données du .json
    """
    jobs_init(data)
    conges_init(data)


def extract_projets(data):
    """
    Extrait les jobs du .json
    """
    jobs = data["jobs"]
    return jobs


def extract_collaborators(data):
    """
    Extrait les données du staff du .json
    """
    staff = data["staff"]
    return staff


def extract_qualifications(data):
    """
    Extrait les données des qualifications du problème du .json
    Note : elles ne sont pas encore liées au staff ici !
    """
    qualifications = data["qualifications"]
    return qualifications


def extract_gain(data):
    """
    Extrait les gains associés aux projets et les organise par ID de projet croissant.
    """
    jobs = data["jobs"]
    gains = []
    for job in jobs:
        gains.append([int(job["name"].replace("Job", "")), job["gain"]])
    gains = [y for x, y in sorted(gains, key=lambda item: item[0])]
    return gains


def extract_penality(data):
    """
    Génère les vecteurs de pénalités associés à chaque projet en fonction du jour (absolu) de terminaison du projet.
    Note : c'est bien jobs["due_date"] et pas (jobs["due_date"] - 1) ici.
    """
    Penalties = []
    for jobs in data["jobs"]:
        row = np.zeros(data["horizon"])
        for i in range(data["horizon"]):
            if i > jobs["due_date"]:
                row[i] = (i - jobs["due_date"]) * jobs["daily_penalty"]
        Penalties.append(row)
    Penalties = np.array(Penalties)
    # print("Pénalités associées aux projets :\n", Penalties)
    return Penalties


def extract_staff_qualifications(data):
    """
    Extrait les qualifications du staff (en ligne : les employés, en colonne : 0/1 selon qu'ils possèdent la qualif.)
    """
    R_qualifications = []
    qualifications = extract_qualifications(data)
    for staff in data["staff"]:
        row = np.zeros(len(qualifications))
        for x in staff["qualifications"]:
            row[qualifications.index(x)] = 1
        R_qualifications.append(row)
    R_qualifications = np.array(R_qualifications)
    # print("\nQualifications des employés :\n", R_qualifications)
    return R_qualifications


def extract_projets_qualifications(data):
    """
    Extrait les classifications demandées par les projets.
    """
    projets_qualifications = []
    qualifications = extract_qualifications(data)
    for job in data["jobs"]:
        row = np.zeros(len(qualifications))
        for x in job["working_days_per_qualification"]:
            row[qualifications.index(x)] = job["working_days_per_qualification"][x]
        projets_qualifications.append(row)
    projets_qualifications = np.array(projets_qualifications)
    # print("\nQualifications des projets :\n", projets_qualifications)
    return projets_qualifications


def extract_staff_conges(data):
    """
    Extrait les jours de congé des employés (en ligne).
    Chaque ligne correspond à un employé, et chaque colonne vaut 0/1 selon que l'employé est en congé le jour correspondant.
    """
    conges = []
    for staff in data["staff"]:
        row = np.zeros(data["horizon"])
        for x in staff["vacations"]:
            row[x] = 1
        conges.append(row)
    conges = np.array(conges)
    # print("\nCongés des employés :\n", conges)
    return conges

def dembedding_project(i):
    return "Job"+str(i+1)

def dembedding_workers(i,data):
    return data["staff"][i]["name"]

def dembedding_competences(i,data):
    qualifications = extract_qualifications(data)
    return qualifications[i]

# *******************************************************************************#
# Tests


def assert_toy_instance(instance):
    """
    Test problem encoding on the toy instance.
    """
    print("\nTesting if the script works on the toy instance.")
    variables = instance.variables
    assert variables["NP"] == 5, "Problem with the number of project"
    assert variables["NC"] == 3, "Problem with the number of workers"
    assert variables["NA"] == 3, "Problem with the number of qualifications"
    assert variables["H"] == 5, "Problem with the horizon"
    assert (
        variables["GAIN"] == np.array([20, 15, 15, 20, 10])
    ).all(), "Problem with the gain"
    assert (
        variables["PENALTIES"]
        == np.array(
            [
                [0, 0, 0, 3, 6],
                [0, 0, 0, 3, 6],
                [0, 0, 0, 0, 3],
                [0, 0, 0, 3, 6],
                [0, 0, 0, 0, 0],
            ]
        )
    ).all(), "Problem with the penalties"
    assert (
        variables["STAFF_QUALIFICATIONS"] == np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
    ).all(), "Problem with the staff qualification"
    assert (
        variables["COST_PROJECT"]
        == np.array([[1, 1, 1], [1, 2, 0], [1, 0, 2], [0, 2, 1], [0, 0, 2]])
    ).all(), "Problem with the cost of project"
    assert (
        variables["CONGES"]
        == np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    ).all(), "Problem with the gain"
    print("All tests passed.")


def assert_medium_instance(instance):
    """
    Test problem encoding on the medium-size instance.
    """
    print("\nTesting if the script works on the medium-size instance.")
    variables = instance.variables
    assert variables["NP"] == 15, "Problem with the number of project"
    assert variables["NC"] == 5, "Problem with the number of workers"
    assert variables["NA"] == 10, "Problem with the number of qualifications"
    assert variables["H"] == 22, "Problem with the horizon"
    assert (
        variables["GAIN"]
        == np.array([15, 30, 30, 30, 70, 40, 20, 10, 25, 20, 25, 45, 40, 60, 50])
    ).all(), "Problem with the gain"
    assert (
        variables["PENALTIES"][:3]
        == np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6],  # 1
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3,
                    6,
                    9,
                    12,
                    15,
                    18,
                ],  # 2
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3,
                    6,
                    9,
                    12,
                    15,
                    18,
                    21,
                    24,
                    27,
                    30,
                ],  # 3
            ]
        )
    ).all(), "Problem with the penalties"
    assert (
        variables["STAFF_QUALIFICATIONS"][0]
        == np.array(
            [
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            ]
        )
    ).all(), "Problem with the staff qualification"
    assert (
        variables["COST_PROJECT"][0] == np.array([[0, 0, 4, 0, 0, 0, 0, 0, 4, 0]])
    ).all(), "Problem with the cost of project"
    assert (
        variables["CONGES"][0]
        == np.array(
            [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
    ).all(), "Problem with the conges"
    print("All tests passed.")


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
            assert self.path, "No path given."
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

    def load_solution(self, model,Affectation,Done_Project,Begin_Project,z_fo2,z_fo3):
        self.model = model
        self.Affectation_solution = Affectation.X
        self.Done_Project_solution = Done_Project.X
        self.Begin_Project_solution = Begin_Project.X
        self.NB_Project_solution = z_fo2.X
        self.Duration_solution = z_fo3.X

    def save_solution(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.write(path + "model.json")
        np.savez("Solution.npz",
                 Affectation_solution=self.Affectation_solution,
                 Done_Project_solution=self.Done_Project_solution,
                 Begin_Project_solution=self.Begin_Project_solution,
                 NB_Project_solution= self.NB_Project_solution,
                 Duration_solution = self.Duration_solution)

        def preprocessing_save(data):
            for worker in data["staff"]:
                rep = []
                for x in worker["vacations"]:
                    rep.append(int(x))
                worker["vacations"] = rep
            data["horizon"] = float(data["horizon"])
            return data

        preprocessing_save(self.data)
        with open(path + "data.json", "w") as write_file:
            json.dump(self.data, write_file, indent=4)

    def load_solution_files(self, path):
        file = open(path + "DATA.json")
        self.data = json.load(file)
        if "model.json" in os.listdir(path):
            file = open(path + "model.json")
            self.model = json.load(file)

    def print_kpi(self):
        assert self.Affectation_solution is not None, "This instance doesn't contain solution data"
        BENEFICES = np.array([self.variables["GAIN"][i] - self.variables["PENALTIES"][i] for i in range(len(self.variables["GAIN"]))])
        project_description = ""
        for project in range(len(self.Done_Project_solution)):
            if np.sum(self.Done_Project_solution[project]) == 1:
                project_description = project_description + \
"The project {} is selected in the solution and done during the day {} and beginned during the day {}.".format(
                    dembedding_project(project),
                    np.where(self.Done_Project_solution[project] == 1)[0][0]+1,
                    np.where(self.Begin_Project_solution[project] == 1)[0][0]+1) + "\n"
            else:
                project_description = project_description + "The project {} isn't selected in the solution.".format(
                    dembedding_project(project)) + "\n"
        gain_text = "The gain is {}.".format(np.sum(self.Done_Project_solution * BENEFICES))
        Max_project_text = "The maximum of project for a worker is {}.".format(self.NB_Project_solution)
        Duration_max_text = "The maximum of duration of a project is {}.".format(self.Duration_solution)
        print(project_description)
        print(gain_text)
        print(Max_project_text)
        print(Duration_max_text)
    def plot_affectation(self):
        assert self.Affectation_solution, "This instance doesn't contain solution data"


# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":

    toy_instance = Instance(path="data/toy_instance.json")
    toy_instance.build_instance()
    assert_toy_instance(toy_instance)
    print("Done.")

    medium_instance = Instance(path="data/medium_instance.json")
    medium_instance.build_instance()
    assert_medium_instance(medium_instance)
    print("Done.")
