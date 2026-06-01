import json
import os

class FunctionDB:
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            with open(path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = []
            with open(path, "w") as f:
                json.dump(self._data, f, indent=4)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=4)

    def add(self, function_arg_dict, function_exec_id):
        data_to_add = {"id": function_exec_id} | function_arg_dict
        self._data.append(data_to_add)
        self.save()

    def _search_indices_by_params(self, function_arg_dict, limit=10):
        indices_to_return = []
        for i, function_data in enumerate(self._data[::-1]):
            is_equal = all([function_data.get(key, None) == val for key, val in function_arg_dict.items()])
            if is_equal: indices_to_return.append(len(self._data) - 1 - i)
            if len(indices_to_return) >= limit: break
        return indices_to_return
    
    def _search_index_by_id(self, id):
        for i, function_data in enumerate(self._data):
            if function_data["id"] == id:
                return i
        raise ValueError("No function execution found with the provided id")
    
    def search_by_params(self, function_arg_dict, qiskit_serverless_service=None, limit=10):
        indices = self._search_indices_by_params(function_arg_dict, limit=limit)
        function_data_to_return = [self._data[i] for i in indices]
        if qiskit_serverless_service is None:
            return function_data_to_return[0] if len(function_data_to_return) == 1 else function_data_to_return
        else:
            jobs = []
            for function_data in function_data_to_return:
                job = qiskit_serverless_service.job(function_data["id"])
                jobs.append(job)
            return jobs[0] if len(jobs) == 1 else jobs
        
    def search_by_id(self, id, qiskit_serverless_service=None):
        data = self._data[self._search_index_by_id(id)]
        if qiskit_serverless_service is None:
            return data
        else:
            job = qiskit_serverless_service.job(data["id"])
            return job
        
    def remove(self, id):
        index = self._search_index_by_id(id)
        self._data.pop(index)
        self.save()

    def update_jobdb(self, jobdb, qiskit_serverless_service, update_all=False):
        for function_data in self._data[::-1]:
            function_result_id = function_data["id"]
            function_job = qiskit_serverless_service.job(function_result_id)
            if function_job.status() == "DONE":
                function_job_result = function_job.result()
                dbdata = function_job_result["dbdata"]
                try:
                    present_dbdata = jobdb.search_by_data(dbdata)
                    if update_all:
                        continue
                    else:
                        break
                except ValueError:
                    jobdb.add_data(dbdata)
            else:
                continue