from z3 import *
import re

class SymbolicVerifier:
    def __init__(self):
        # We use a persistent solver state for the duration of one reasoning task
        self.solver = Solver()
        self.variables = {}

    def _get_var(self, name):
        """Helper to manage dynamic boolean variables."""
        name = re.sub(r'\W+', '', name) # Sanitize
        if name not in self.variables:
            self.variables[name] = Bool(name)
        return self.variables[name]

    def add_premise(self, premise_text: str):
        """
        In a full paper implementation, an NLP-to-Logic parser would go here.
        For the prototype, we assume the agent outputs: 'Fact: VariableName'
        """
        if "Fact:" in premise_text:
            var_name = premise_text.split("Fact:")[1].strip()
            self.solver.add(self._get_var(var_name) == True)

    def check_consistency(self, step_text: str) -> bool:
        """
        Checks if the 'step_text' is logically possible given previous facts.
        """
        # Simple Logic Extraction (e.g., 'Step: not A')
        if "not" in step_text.lower():
            # Example: "Therefore, not A"
            match = re.search(r'not\s+(\w+)', step_text)
            if match:
                var_name = match.group(1)
                var = self._get_var(var_name)
                
                self.solver.push() # Check state
                self.solver.add(var == False)
                result = self.solver.check()
                self.solver.pop() # Restore state
                
                return result == sat # Returns False if 'not A' is impossible (unsat)
        
        return True # Default to True for non-logical prose steps