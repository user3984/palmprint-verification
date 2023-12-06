import json
import os
from typing import Optional, Tuple, List
import cv2 as cv
import numpy as np
import pickle

from backend.api import try_get_roi, roi_to_embeddings


T = 0.5


class Service:
    def __init__(self, db_path=None) -> None:
        self.db_path = db_path
        self.redundancy = 10
        self.embs = self._load_db(db_path)
        # self.embs = []

    def _load_db(self, db_path) -> List[Tuple[List, str]]:
        if db_path is None:
            return []
        # with open(db_path, "r") as fp:
        #     return json.load(fp)
        with open(db_path, 'rb') as fp:
            return pickle.load(fp)

    def demo_roi_camera(self) -> None:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
        while True:
            ret, frame = cap.read()
            if ret:
                display = frame
                roi = try_get_roi(frame)
                if roi is not None:
                    display = roi
                cv.imshow('Camera Feed', display)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        cv.waitKey(1)

    def verify_id(self) -> Optional[str]:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
        user = None
        while True:
            ret, frame = cap.read()
            if ret:
                display = frame
                roi = try_get_roi(frame)
                if roi is not None:
                    display = roi
                    user = self._pp_in_db(roi)
                if user is not None:
                    break
                cv.imshow('Camera Feed', display)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        cv.waitKey(1)
        return user

    def _pp_in_db(self, roi) -> Optional[str]:
        if len(self.embs) == 0:
            return None

        emb = roi_to_embeddings(roi)
        sims = [(np.dot(emb, e), u) for e, u in self.embs]
        s, u = sorted(sims, reverse=True)[0]
        if s < T:
            return None
        return u

    def user_exists(self, name) -> bool:
        return name.strip() in self.users

    def add_user(self, name) -> None:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        embs = []
        while True and len(embs) < self.redundancy:
            ret, frame = cap.read()
            if ret:
                display = frame
                roi = try_get_roi(frame)
                if roi is not None:
                    display = roi
                    e = roi_to_embeddings(roi)
                    if len(e):
                        embs.append(e)
                cv.imshow('Camera Feed', display)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        cv.waitKey(1)

        if len(embs) == self.redundancy:
            for e in embs:
                self.embs.append((e, name))
            print(f"User {name} added to database.")
        elif len(embs) < self.redundancy:
            print(f"Failed to add user {name} to database.")
        else:
            raise

    def save_db(self, fname) -> None:
        # path = os.path.join("store", f"{fname}.json")
        # with open(path, "w") as fp:
        #     json.dump(self.embs, fp)
        path = os.path.join("store", f"{fname}.pkl")
        with open(path, 'wb') as file:
            pickle.dump(self.embs, file)

    @property
    def users(self):
        return set([u for _, u in self.embs])
