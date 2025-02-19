import argparse
import uuid
from dataclasses import dataclass
from random import randint, sample, choice
from typing import List
import json
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from util.language_model import LanguageModel

"""
Generate synthetic archetypes and actions for archetypes
"""


DATA_PATH = "./data/synthetic_data_tasks"

STREAMING = "Streaming Videos"
SOCIAL_MEDIA = "Browse Social Media"
SHOPPING = "Online Shopping"
EMAIL = "Online research"
READING = "Reading Articles"
WORK_APPS = "Using web based work applications"
FINANCE = "Online banking and Financial Management"
FORUMS = "Participating in online forums and communities"
TRAVEL = "Researching or Booking Travel"
HOBBY = "Exploring a hobby online"

TASKS = [
    STREAMING,
    SOCIAL_MEDIA,
    SHOPPING,
    EMAIL,
    READING,
    WORK_APPS,
    FINANCE,
    FORUMS,
    TRAVEL,
]

SUBTASKS = {
    STREAMING: ["news_subtopics.json"],
    SOCIAL_MEDIA: [],
    SHOPPING: ["shopping_subtopics.json"],
    EMAIL: ["email_subtopics.json"],
    READING: ["news_subtopics.json"],
    WORK_APPS: [],
    FINANCE: [],
    FORUMS: ["news_subtopics.json"],
    TRAVEL: ["travel_subtopics.json"],
    HOBBY: []
}

OPEN_TABS_PER_TASK = {
    "default": {
        "min": 1,
        "max": 10
    }
}

TYPICAL_TASK_COUNTS = [1, 2, 3, 4, 5, 9, 10]

TAB_URL_KEY = "url"
TAB_TITLE_KEY = "title"

@dataclass
class UserArchetype:
    user_description: str = None
    task_list: str = None


class TabGroupingDataGenerator:
    all_archetypes: List[UserArchetype] = []

    def __init__(self):
        self.lm = LanguageModel()

    def load_subtopic_data(self):
        with open(os.path.join(DATA_PATH, "news_subtopics.json")) as json_file:
            self.news_subtopics = json.load(json_file)['subtopics']
        with open(os.path.join(DATA_PATH, "shopping_subtopics.json")) as json_file:
            self.shopping_subtopics = json.load(json_file)['subtopics']
        with open(os.path.join(DATA_PATH, "email_subtopics.json")) as json_file:
            self.email_subtopics = json.load(json_file)['subtopics']
        with open(os.path.join(DATA_PATH, "travel_subtopics.json")) as json_file:
            self.travel_subtopics = json.load(json_file)['subtopics']
        with open(os.path.join(DATA_PATH, "user_locations.json")) as json_file:
            self.user_locations = json.load(json_file)

    def get_subtopics_with_prefixes(self, task):
        if task in [STREAMING, READING, FORUMS]:
            return self.news_subtopics, "about", 3
        elif task in [SHOPPING]:
            return self.shopping_subtopics, "for", 2
        elif task in [EMAIL]:
            return self.email_subtopics, "for", 1
        elif task in [TRAVEL]:
            return self.travel_subtopics, "for", 1
        return [], None

    def generate_tasks_with_subtopics(self, tasks):
        generated_tasks = []
        for task in tasks:
            subtask_supported = len(SUBTASKS.get(task, [])) > 0
            if subtask_supported:
                data, prefix, sample_size = self.get_subtopics_with_prefixes(task)
                sampled_subtasks = sample(data, sample_size)
                new_tasks = [f"{task} {prefix} {l}" for l in sampled_subtasks]
                generated_tasks += new_tasks
            else:
                generated_tasks += [task]
        return list(set(generated_tasks))

    def compute_tasks_for_user(self, archetype_name, task_list):
        ls = self.generate_tasks_with_subtopics(task_list)
        task_name = f"We have a web user of the following archetype. {archetype_name}. Indicate whether they would typically do the following task in a desktop web browser."
        tasks_supported = self.lm.ask_list_boolean(ls.copy(), task_name)
        tasks_df = pd.DataFrame({"tasks_supported": tasks_supported, "task": ls.copy()})
        tasks_for_user = tasks_df[tasks_df.tasks_supported].task.to_list()
        return tasks_for_user

    def build_archetypes(self, n_archetypes=10, add_location=False):
        self.load_subtopic_data()
        archetypes = self.lm.get_list(
            f"Create a list of {n_archetypes} of the user archetypes who use desktop web browsers. Include their gender, age, and other relevant info that might determine what sites they browse")
        for arch in archetypes:
            location = choice(self.user_locations)
            arch = f"{arch} Located in {location}."
            print(arch)
            tasks_for_user = self.compute_tasks_for_user(arch, TASKS)
            user = UserArchetype(user_description=arch, task_list=tasks_for_user)
            self.all_archetypes.append(user)

    def gen_data_for_archetype(self, user: UserArchetype, allow_same_task_twice=False, n_sample_pages=10, retarget_topic=True):
        print(f"gen_data_for_archetype {n_sample_pages}")
        data_list = []
        task_count_index = randint(0, len(TYPICAL_TASK_COUNTS) - 1)
        task_count = TYPICAL_TASK_COUNTS[task_count_index]
        test_set_id = str(uuid.uuid4())
        task_list = self.compute_tasks_for_user(user.user_description, TASKS)
        cur_task_list = pd.Series(task_list).sample(n=min(task_count, len(task_list))).to_list()
        for task in cur_task_list:
            print(f"running task {task}")
#            if randrange(2) or True:
#                revised_task = self.lm.text_query(f"I'm trying to define a task for the following user archetype: {user.user_description}. Please update the following task to make it more specific for an imagined scenario that you dream up. Use just a few words ### {task}", retry_count=1)
#                if revised_task is not None and len(revised_task) > 0:
#                    print(f"Made task more specific as {revised_task}")
#                    task = revised_task
            browsing_data = self.lm.ask_df(
                f"We are generating sample browsing data for a user of the following user. {user.user_description}"
                f"Generate {n_sample_pages} sample page tiles and URLs for the user performing a specific instance task {task} in a single browser session",
                [TAB_URL_KEY, TAB_TITLE_KEY])
            open_tabs_info = OPEN_TABS_PER_TASK[task] if task in OPEN_TABS_PER_TASK else OPEN_TABS_PER_TASK["default"]
            num_open_tabs = randint(open_tabs_info["min"], open_tabs_info["max"])
            browsing_data = browsing_data.sample(n=num_open_tabs).reset_index(drop=True)
            browsing_data["task"] = task
            browsing_data["test_set_id"] = test_set_id
            browsing_data["task_id"] = f"{test_set_id}_{str(uuid.uuid4())}"
            browsing_data["user_description"] = user.user_description
            data_list.append(browsing_data)
        return pd.concat(data_list, axis=0)

    def build_data(self, num_tests_per_user: int = 40, archetypes=None):
        data_list = []
        if archetypes is None:
            archetypes = self.all_archetypes
        for arch in archetypes:
            print(f"working on archetype {arch.user_description}")
            for i in range(num_tests_per_user):
                try:
                    data_list.append(self.gen_data_for_archetype(arch))
                except Exception as ex:
                    print(f"Failed instance {i} archetype {arch}: {ex}")
        return pd.concat(data_list, axis=0)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generates synthetic data for tab grouping")
    parser.add_argument('--num_archetypes', type=int, default=50, help='Number of archetypes')
    parser.add_argument('--num_tests_per_user', type=int, default=40, help='Number of tests for a user archetype')
    args = parser.parse_args()

    gen = TabGroupingDataGenerator()
    print(f"Generating {args.num_archetypes} archetypes")
    gen.build_archetypes(n_archetypes=args.num_archetypes, add_location=True)
    cur_data = pd.DataFrame()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_filename = f"output/gen_data_{timestamp}.csv"
    # we get about 5 clusters per user, so 20 arch * 50 users = 5000 clusters
    for archetype in gen.all_archetypes:
        print(f"Generating {args.num_tests_per_user} tests for {archetype.user_description}")
        data = gen.build_data(num_tests_per_user=args.num_tests_per_user, archetypes=[archetype])
        cur_data = pd.concat([cur_data, data], axis=0)
        print(f"finished arch {archetype.user_description}. Saving in progress file")
        cur_data.to_csv(out_filename, index=False)
    print(f"All done. Final result in {out_filename}")

