from msal import PublicClientApplication

# Different client ID that works with personal accounts
CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"  # Azure CLI client
AUTHORITY = "https://login.microsoftonline.com/common"

SCOPES = [
    "https://graph.microsoft.com/Files.Read.All",
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/Calendars.ReadWrite",
    "https://graph.microsoft.com/OnlineMeetings.ReadWrite"
]

app = PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

print("Opening browser for sign-in...")
print("A browser window will pop up - sign in with dhrutipurushotham@gmail.com")

try:
    result = app.acquire_token_interactive(
        scopes=SCOPES,
        prompt="select_account"  # Let you choose which account
    )
    
    if result and "access_token" in result:
        token = result["access_token"]
        
        print("\n" + "="*50)
        print("✅ SUCCESS!")
        print("="*50)
        print(f"Token length: {len(token)}")
        print(f"Starts with: {token[:10]}")
        print(f"Is JWT: {token.startswith('eyJ')}")
        
        print("\n📋 Copy this to .env:")
        print("="*50)
        print(f"MS_GRAPH_ACCESS_TOKEN={token}")
        print("="*50)
        
        # Save to file
        with open("token.txt", "w") as f:
            f.write(token)
        print("\n✅ Also saved to token.txt")
        
    else:
        print(f"❌ Error: {result}")
        
except Exception as e:
    print(f"❌ Exception: {e}")
    print("\nTrying alternative method...")