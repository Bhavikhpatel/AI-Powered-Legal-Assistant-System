from api.app import app

# Vercel expects 'app' variable
if __name__ == '__main__':
    app.run(debug=True)
