from app.services.google_drive import run_oauth_flow

if __name__ == "__main__":
    run_oauth_flow()
    print("Google authorization saved. You can now start the backend.")
