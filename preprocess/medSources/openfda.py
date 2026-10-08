import requests


OPENFDA_URL = "https://api.fda.gov/drug/label.json"


class OpenFDASource:

    def fetch_drug(self, generic_name: str):

        params = {
            "search": f'openfda.generic_name:"{generic_name}"',
            "limit": 400,
        }

        response = requests.get(
            OPENFDA_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

if __name__ == "__main__":

    source = OpenFDASource()

    data = source.fetch_drug("METFORMIN")

    print(
        f"Found {len(data.get('results', []))} records"
    )