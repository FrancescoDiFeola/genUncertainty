class LatentMemory:
    """Stores latent representations for Diff_2000 and Diff_2001, delaying loss computation until fully populated."""
    def __init__(self):
        self.memory = {}  # Dictionary to store {patient_id: {"Diff_2000": tensor, "Diff_2001": tensor}}
        self.filled = False  # Flag to indicate when the memory is fully populated

    def update(self, patient_id, diff_type, latent_vector):
        """Update the memory bank with the latest latent vector."""
        if patient_id not in self.memory:
            self.memory[patient_id] = {"Diff_2000": None, "Diff_2001": None}

        # Replace the old latent with the new one
        self.memory[patient_id][diff_type] = latent_vector

    def get_pair(self, patient_id, diff_type):
        """Retrieve the corresponding latent from the memory bank."""
        opposite_type = "Diff_2000" if diff_type == "Diff_2001" else "Diff_2001"
        return self.memory.get(patient_id, {}).get(opposite_type, None)

    def check_filled(self):
        """Check if the memory bank is fully populated with at least one latent per patient."""
        for patient_id, data in self.memory.items():
            if data["Diff_2000"] is None or data["Diff_2001"] is None:
                return False  # Memory is not fully populated yet
        self.filled = True  # Mark as fully populated
        return True