# from pymongo import MongoClient
# from pymongo.errors import ConnectionFailure, OperationFailure
# from urllib.parse import quote_plus  

# # Global MongoDB client and collections
# mongo_client = None
# db = None
# _collection = None
# _collection1 = None
# _collection2 = None
# _collection3 = None
# _collection4 = None
# _collection5 = None
# _collection6 = None
# _collection7 = None
# _collection8 = None
# _collection9 = None
# _collection10 = None
# mongo_credentials = {
#     "connection_string": None,
#     "password": None,
#     "db_name": None
# }

# def set_mongo_credentials(connection_string, password, db_name):
#     """Store MongoDB credentials globally."""
#     mongo_credentials["connection_string"] = connection_string
#     mongo_credentials["password"] = password
#     mongo_credentials["db_name"] = db_name
    
# def get_mongo_client(connection_string: str, password: str, db_name: str):
#     """Initialize and return a MongoDB client connection dynamically."""
#     global mongo_client, db, _collection, _collection1, _collection2,_collection3,_collection4,_collection5,_collection6,_collection7,_collection8,_collection9,_collection10 

#     try:
#         # Ensure password is URL-encoded
#         encoded_password = quote_plus(password)

#         # Replace placeholder with the actual password
#         mongo_uri = connection_string.replace("<db_password>", encoded_password)

#         # Create a new client
#         new_client = MongoClient(mongo_uri)

#         # Verify authentication
#         try:
#             new_client.admin.command("ping")  
#         except OperationFailure as auth_error:
#             raise Exception("Authentication failed! Check your username and password.") from auth_error

#         # Assign new connection if successful
#         mongo_client = new_client
#         db = mongo_client[db_name]

#         # Create collections if they do not exist
#         _collection = db["product"]
#         _collection1 = db["crate_count"]
#         _collection2 = db["milk_spillage"]
#         _collection3 = db["milk_wastage"]
#         _collection4 = db["conveyor_belt_crate_count"]
#         _collection5 = db["Speed_Monitoring"]
#         _collection6 = db["helmet_safety"]
#         _collection7 = db["safety_detection"]
#         _collection8 = db["Intrusion_detection"]
#         _collection9 = db["crowd_monitoring"]
#         _collection10 = db["camera_tampering"]

#         return mongo_client, db  

#     except ConnectionFailure:
#         raise Exception("Failed to connect to MongoDB. Check your network or MongoDB URI.")
#     except Exception as e:
#         raise Exception(f"Error: {e}")

# def set_mongo_client(client, database):
#     """Explicitly set the global MongoDB client and database after reconnection or disconnection."""
#     global mongo_client, db, _collection, _collection1, _collection2,_collection3,_collection4,_collection5,_collection6,_collection7,_collection8,_collection9,_collection10

#     mongo_client = client
#     db = database

#     if db is not None:  
#         _collection = db["product"]
#         _collection1 = db["crate_count"]
#         _collection2 = db["milk_spillage"]
#         _collection3 = db["milk_wastage"]
#         _collection4 = db["conveyor_belt_crate_count"]
#         _collection5 = db["Speed_Monitoring"]
#         _collection6 = db["helmet_safety"]
#         _collection7 = db["safety_detection"]
#         _collection8 = db["Intrusion_detection"]
#         _collection9 = db["crowd_monitoring"]
#         _collection10 = db["camera_tampering"]
#     else:
#         _collection = None
#         _collection1 = None
#         _collection2 = None
#         _collection3 = None
#         _collection4 = None
#         _collection5 = None
#         _collection6 = None
#         _collection7 = None
#         _collection8 = None
#         _collection9 = None
#         _collection10 = None

# def get_collection(): 
#     """Return the 'product' collection if initialized."""
#     if _collection is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection

# def get_collection1(): 
#     """Return the 'crate_count' collection if initialized."""
#     if _collection1 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection1

# def get_collection2(): 
#     """Return the 'milk_spillage' collection if initialized."""
#     if _collection2 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection2

# def get_collection3(): 
#     """Return the 'milk_wastage' collection if initialized."""
#     if _collection3 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection3

# def get_collection4(): 
#     """Return the 'conveyor_belt_crate_count' collection if initialized."""
#     if _collection4 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection4

# def get_collection5(): 
#     """Return the 'Speed_Monitoring' collection if initialized."""
#     if _collection5 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection5

# def get_collection6(): 
#     """Return the 'helmet_safety' collection if initialized."""
#     if _collection6 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection6

# def get_collection7(): 
#     """Return the 'safety_detection' collection if initialized."""
#     if _collection7 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection7

# def get_collection8(): 
#     """Return the 'Intrusion_detection' collection if initialized."""
#     if _collection8 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection8

# def get_collection9(): 
#     """Return the 'crowd_monitoring' collection if initialized."""
#     if _collection9 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection9

# def get_collection10(): 
#     """Return the 'camera_tampering' collection if initialized."""
#     if _collection10 is None:
#         raise Exception("Database not initialized. Connect first.")
#     return _collection10


from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from urllib.parse import quote_plus  

# Global MongoDB client and collections
mongo_client = None
db = None
_collection = None
_collection1 = None
_collection2 = None
_collection3 = None
_collection4 = None
_collection5 = None
_collection6 = None
_collection7 = None
_collection8 = None
_collection9 = None
_collection10 = None
_collection_cam_details = None
mongo_credentials = {
    "connection_string": None,
    "password": None,
    "db_name": None
}

def set_mongo_credentials(connection_string, password, db_name):
    """Store MongoDB credentials globally."""
    mongo_credentials["connection_string"] = connection_string
    mongo_credentials["password"] = password
    mongo_credentials["db_name"] = db_name
    
def get_mongo_client(connection_string: str, password: str, db_name: str):
    """Initialize and return a MongoDB client connection dynamically."""
    global mongo_client, db, _collection, _collection1, _collection2, _collection3, _collection4, _collection5, _collection6, _collection7, _collection8, _collection9, _collection10, _collection_cam_details

    try:
        # Ensure password is URL-encoded
        encoded_password = quote_plus(password)

        # Replace placeholder with the actual password
        mongo_uri = connection_string.replace("<db_password>", encoded_password)

        # Create a new client
        new_client = MongoClient(mongo_uri)

        # Verify authentication
        try:
            new_client.admin.command("ping")  
        except OperationFailure as auth_error:
            raise Exception("Authentication failed! Check your username and password.") from auth_error

        # Assign new connection if successful
        mongo_client = new_client
        db = mongo_client[db_name]

        # Create collections if they do not exist
        _collection = db["product"]
        _collection1 = db["crate_count"]
        _collection2 = db["milk_spillage"]
        _collection3 = db["milk_wastage"]
        _collection4 = db["conveyor_belt_crate_count"]
        _collection5 = db["Speed_Monitoring"]
        _collection6 = db["helmet_safety"]
        _collection7 = db["safety_detection"]
        _collection8 = db["Intrusion_detection"]
        _collection9 = db["crowd_monitoring"]
        _collection10 = db["camera_tampering"]
        _collection_cam_details = db["cam_details"]

        return mongo_client, db  

    except ConnectionFailure:
        raise Exception("Failed to connect to MongoDB. Check your network or MongoDB URI.")
    except Exception as e:
        raise Exception(f"Error: {e}")

def set_mongo_client(client, database):
    """Explicitly set the global MongoDB client and database after reconnection or disconnection."""
    global mongo_client, db, _collection, _collection1, _collection2, _collection3, _collection4, _collection5, _collection6, _collection7, _collection8, _collection9, _collection10, _collection_cam_details

    mongo_client = client
    db = database

    if db is not None:  
        _collection = db["product"]
        _collection1 = db["crate_count"]
        _collection2 = db["milk_spillage"]
        _collection3 = db["milk_wastage"]
        _collection4 = db["conveyor_belt_crate_count"]
        _collection5 = db["Speed_Monitoring"]
        _collection6 = db["helmet_safety"]
        _collection7 = db["safety_detection"]
        _collection8 = db["Intrusion_detection"]
        _collection9 = db["crowd_monitoring"]
        _collection10 = db["camera_tampering"]
        _collection_cam_details = db["cam_details"]
    else:
        _collection = None
        _collection1 = None
        _collection2 = None
        _collection3 = None
        _collection4 = None
        _collection5 = None
        _collection6 = None
        _collection7 = None
        _collection8 = None
        _collection9 = None
        _collection10 = None
        _collection_cam_details = None

def get_collection(): 
    """Return the 'product' collection if initialized."""
    if _collection is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection

def get_collection1(): 
    """Return the 'crate_count' collection if initialized."""
    if _collection1 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection1

def get_collection2(): 
    """Return the 'milk_spillage' collection if initialized."""
    if _collection2 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection2

def get_collection3(): 
    """Return the 'milk_wastage' collection if initialized."""
    if _collection3 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection3

def get_collection4(): 
    """Return the 'conveyor_belt_crate_count' collection if initialized."""
    if _collection4 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection4

def get_collection5(): 
    """Return the 'Speed_Monitoring' collection if initialized."""
    if _collection5 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection5

def get_collection6(): 
    """Return the 'helmet_safety' collection if initialized."""
    if _collection6 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection6

def get_collection7(): 
    """Return the 'safety_detection' collection if initialized."""
    if _collection7 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection7

def get_collection8(): 
    """Return the 'Intrusion_detection' collection if initialized."""
    if _collection8 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection8

def get_collection9(): 
    """Return the 'crowd_monitoring' collection if initialized."""
    if _collection9 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection9

def get_collection10(): 
    """Return the 'camera_tampering' collection if initialized."""
    if _collection10 is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection10

def get_collection_cam_details(): 
    """Return the 'cam_details' collection if initialized."""
    if _collection_cam_details is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection_cam_details

def save_cam(camera_name: str, location: str, stream_url: str):
    """Save camera details to the 'cam_details' collection."""
    try:
        collection = get_collection_cam_details()
        camera_data = {
            "camera_name": camera_name,
            "location": location,
            "stream_url": stream_url
        }
        result = collection.insert_one(camera_data)
        return result.inserted_id
    except OperationFailure as e:
        raise Exception(f"Failed to save camera details: {str(e)}")
    except Exception as e:
        raise Exception(f"Error saving camera details: {str(e)}")