

def get_claims(credential, scope="https://management.azure.com/.default"):
    import base64, json

    print("Validicating credential....")
    # Acquire token for the given scope
    token_str = credential.get_token(scope).token


    parts = token_str.split(".")
    if len(parts) < 2:
        raise ValueError("Invalid token format.")
        
    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)  # pad properly
    payload_json = base64.urlsafe_b64decode(payload_b64)
    claims = json.loads(payload_json)
    print("Credential Validated")
    return claims

def get_unique_name(credential, scope="https://management.azure.com/.default"):

    claims = get_claims(credential, scope)
    print("Determining unique name....")
    name = claims.get("unique_name")
    unique_name = name.split('@')[0]
    print(f"Unique name: {unique_name}")
    return unique_name

