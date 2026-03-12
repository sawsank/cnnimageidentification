from deepface import DeepFace
import os
import shutil
import cv2
import pandas as pd

class FaceEngine:
    def __init__(self, db_path="database"):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
    def enroll_face(self, image_path, name):
        """
        Save an image to the database folder with the person's name.
        """
        # Ensure name is filesystem friendly
        clean_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).strip()
        target_path = os.path.join(self.db_path, f"{clean_name}.jpg")
        
        # If file exists, we could append a number, but for simplicity we'll overwrite or assume unique names
        shutil.copy(image_path, target_path)
        return target_path

    def recognize_face(self, img_path):
        """
        Recognize a face from an image path by comparing against the database.
        """
        try:
            # DeepFace.find returns a list of dataframes
            results = DeepFace.find(img_path=img_path, 
                                   db_path=self.db_path, 
                                   model_name="VGG-Face", 
                                   enforce_detection=False,
                                   silent=True)
            
            if len(results) > 0 and not results[0].empty:
                # The first result in the first dataframe is the best match
                best_match = results[0].iloc[0]
                identity = best_match['identity']
                
                # Dynamically find the distance column (usually {model_name}_cosine)
                distance_cols = [col for col in best_match.index if "cosine" in col.lower()]
                distance = best_match[distance_cols[0]] if distance_cols else 0.0
                
                # Extract name from filename
                name = os.path.splitext(os.path.basename(identity))[0]
                return {"name": name, "distance": float(distance), "found": True}
            else:
                return {"name": "Unknown", "distance": None, "found": False}
        except Exception as e:
            # Log the columns to help debug if it fails again
            import traceback
            print(f"Error in recognition: {e}")
            return {"name": "Error", "message": str(e), "found": False}

if __name__ == "__main__":
    # Quick test
    engine = FaceEngine()
    print("Testing recognition on Shasank...")
    res = engine.recognize_face("database/shasank.jpg")
    print(res)
