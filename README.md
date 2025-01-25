# botlab-models
RESTful APIs for use at the Robotics and IoT lab at BITS Pilani.

docker build -t botlab-models .

docker run -d -p 5001:5001 --name botlab botlab-models 

Sample curl request-

 curl --location --request POST 'http://localhost:5000/invocations' --header 'Content-Type: application/json' --data-raw '{
    "images": [
        "https://images.unsplash.com/photo-1597308680537-1ba44407ffc0?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1834&q=80",
        "https://images.unsplash.com/photo-1589270216117-7972b3082c7d?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1834&q=80"],
    "classes": ["person", "bag", "person with a bag", "woman riding a horse", "woman with a bag", "woman with black shirt and a bag"]
}'