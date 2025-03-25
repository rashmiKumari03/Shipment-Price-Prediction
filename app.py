# Import FastAPI and related modules for creating API endpoints and handling HTTP requests
from fastapi import FastAPI,Request, HTTPException, Form, Depends 
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sys
import uvicorn
import os 


# Import custom modules for model prediction, training, and logging
from src.Shipment_Price_Prediction.utils.main_utils import MainUtils
from src.Shipment_Price_Prediction.components.model_predictor import CostPredictor, ShippingData
from src.Shipment_Price_Prediction.constant import APP_HOST, APP_PORT
from src.Shipment_Price_Prediction.pipelines.training_pipeline import TrainPipeline
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

import warnings
warnings.filterwarnings("ignore")

# Initialize FastAPI application
logging.info("Starting FastAPI application...")
app = FastAPI()

# Serve static files like CSS, JS, images
logging.info("Configuring static file mount at '/static'...")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates for HTML rendering
logging.info("Setting up Jinja2 templates...")
templates = Jinja2Templates(directory="templates")

# Configure CORS to allow all origins for testing
logging.info("Enabling CORS for all origins...")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Class to parse and store form data submitted by users
logging.info("Defining the DataForm class to manage user form submissions...")
class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        
        # Identification and Basic Information
        self.country: Optional[str] = None
        self.vendor: Optional[str] = None
        self.molecule_test_type: Optional[str] = None
        self.brand: Optional[str] = None
        self.dosage: Optional[str] = None
        self.dosage_form: Optional[str] = None
        self.manufacturing_site: Optional[str] = None
        
        # Shipping Information
        self.line_item_quantity: Optional[int] = None
        self.fulfill_via: Optional[str] = None
        self.vendor_inco_term: Optional[str] = None
        self.shipment_mode: Optional[str] = None
        self.scheduled_delivery_date: Optional[str] = None
        self.delivered_to_client_date: Optional[str] = None
        self.delivery_recorded_date: Optional[str] = None
        
        # Financial Information
        self.line_item_value: Optional[float] = None
        self.pack_price: Optional[float] = None
        self.unit_price: Optional[float] = None
        self.weight: Optional[float] = None
        self.freight_cost: Optional[float] = None
        self.line_item_insurance: Optional[float] = None
        self.unit_of_measure: Optional[float] = None
        
        # Additional Information
        self.first_line_designation: Optional[int] = None
        self.product_group: Optional[str] = None
        self.sub_classification: Optional[str] = None

    async def get_shipping_data(self):
        logging.info("Extracting shipping data from form...")
        form = await self.request.form()
        
        # Identification and Basic Information
        self.country = form.get("country", "")
        self.vendor = form.get("vendor", "")
        self.molecule_test_type = form.get("molecule_test_type", "")
        self.brand = form.get("brand", "")
        self.dosage = form.get("dosage", "")
        self.dosage_form = form.get("dosage_form", "")
        self.manufacturing_site = form.get("manufacturing_site", "")
        
        # Shipping Information
        self.line_item_quantity = int(form.get("line_item_quantity", 0))
        self.fulfill_via = form.get("fulfill_via", "")
        self.vendor_inco_term = form.get("vendor_inco_term", "")
        self.shipment_mode = form.get("shipment_mode", "")
        self.scheduled_delivery_date = form.get("scheduled_delivery_date", "")
        self.delivered_to_client_date = form.get("delivered_to_client_date", "")
        self.delivery_recorded_date = form.get("delivery_recorded_date", "")
        
        # Financial Information
        self.line_item_value = float(form.get("line_item_value", 0.0))
        self.pack_price = float(form.get("pack_price", 0.0))
        self.unit_price = float(form.get("unit_price", 0.0))
        self.weight = float(form.get("weight", 0.0))
        self.freight_cost = float(form.get("freight_cost", 0.0))
        self.line_item_insurance = float(form.get("line_item_insurance", 0.0))
        self.unit_of_measure = float(form.get("unit_of_measure", 0.0))
        
        # Additional Information
        self.first_line_designation = int(form.get("first_line_designation", 0))
        self.product_group = form.get("product_group", "")
        self.sub_classification = form.get("sub_classification", "")
        

        
# -------------------- ROUTES -------------------- #
   
# Homepage route
logging.info("Adding homepage route at '/'...")
@app.get("/")
async def home(request: Request):
    logging.info("Rendering Home_page.html...")
    return templates.TemplateResponse("Home_page.html", {"request": request})

# Price prediction form page
logging.info("Adding price prediction route at '/price-prediction'...")
@app.get("/price-prediction")
async def price_prediction(request: Request):
    logging.info("Rendering price_prediction.html...")
    return templates.TemplateResponse("price_prediction.html", {"request": request})

# Dashboard page
logging.info("Adding dashboard route at '/dashboard'...")
@app.get("/dashboard")
async def dashboard(request: Request):
    logging.info("Rendering dashboard_page.html...")
    return templates.TemplateResponse("dashboard_page.html", {"request": request})

# KPI page
logging.info("Adding KPI route at '/kpi'...")
@app.get("/kpi")
async def kpi(request: Request):
    logging.info("Rendering kpi_page.html...")
    return templates.TemplateResponse("kpi_page.html", {"request": request})



# Fetch developer credentials from .env (I have already initalized the username and password in .env (as envirnoment variable))
# Clean up any quotes or whitespace around the env variables
DEVELOPER_USERNAME = os.getenv("DEVELOPER_USERNAME", "").strip().strip("'").strip('"')
DEVELOPER_PASSWORD = os.getenv("DEVELOPER_PASSWORD", "").strip().strip("'").strip('"')

logging.info("Loaded DEVELOPER_USERNAME: %s", repr(DEVELOPER_USERNAME))
logging.info("Loaded DEVELOPER_PASSWORD: %s", repr(DEVELOPER_PASSWORD))


# Session storage for simple login flow (for demo purposes)
logged_in_users = set()


# -------------------- DEVELOPER ROUTES -------------------- #
logging.info("Configuring routes for model training...")

@app.get("/developer/train-model")
async def developer_check(request: Request):
    """
    GET /developer/train-model
    - Displays a page asking if the user is a developer (developer_check.html)
    """
    logging.info("Accessing 'Are you a developer?' check page.")
    return templates.TemplateResponse("developer_check.html", {"request": request})


@app.post("/developer/train-model")
async def verify_developer(request: Request, is_developer: str = Form(...)):
    """
    POST /developer/train-model
    - Processes the developer confirmation form
    - Redirects to login if 'yes', reloads the page if 'no'
    """
    logging.info("Processing developer verification form.")
    if is_developer == "yes":
        logging.info("User confirmed as developer. Redirecting to login page.")
        return RedirectResponse(url="/developer/train-model/login", status_code=303)
    logging.info("User denied being a developer. Reloading check page.")
    return templates.TemplateResponse("developer_check.html", {"request": request})

# Developer login routes
@app.get("/developer/train-model/login")
async def show_login_form(request: Request):
    logging.info("Rendering developer login form.")
    return templates.TemplateResponse("developer_check.html", {"request": request, "login_error": None})

@app.post("/developer/train-model/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    logging.info(f"Processing login attempt for user: {username}")

    if username.strip() == DEVELOPER_USERNAME and password.strip() == DEVELOPER_PASSWORD:
        logged_in_users.add(request.client.host)
        logging.info(f"Login successful for {username} from IP {request.client.host}")
        return JSONResponse(content={"status": 200, "message": "Login successful!"})
    
    logging.warning(f"Invalid login attempt for user: {username}")
    return JSONResponse(content={"status": 401, "message": "Invalid credentials."}, status_code=401)

# Model training confirmation routes
@app.get("/developer/train-model/confirm")
async def confirm_training(request: Request):
    client_ip = request.client.host
    logging.info(f"Accessing training confirmation page from IP: {client_ip}")
    if client_ip not in logged_in_users:
        logging.warning(f"Unauthorized access attempt to confirmation page from IP: {client_ip}")
        return RedirectResponse(url="/developer/train-model/login", status_code=303)

    logging.info("Rendering model training confirmation page.")
    return templates.TemplateResponse("developer_check.html", {"request": request, "message": "Do you want to start model training?"})


@app.post("/developer/train-model/confirm")
async def process_training_confirmation(request: Request, confirm: str = Form(...)):
    client_ip = request.client.host
    logging.info(f"Processing model training confirmation from IP: {client_ip}")

    # Authentication check
    if client_ip not in logged_in_users:
        logging.warning(f"Unauthorized training confirmation attempt from IP: {client_ip}")
        return RedirectResponse(url="/developer/train-model/login", status_code=303)

    # Process user confirmation
    if confirm.lower() == "yes":
        logging.info("User confirmed model training.")
        return RedirectResponse(url="/developer/train-model/trainingstarted", status_code=303)
    else:
        logging.info("User declined model training.")
        return templates.TemplateResponse(
            "developer_check.html", 
            {"request": request, "message": "Model training was not started."}
        )
        
        
# Model training initiation route
@app.get("/developer/train-model/trainingstarted")
async def training_started(request: Request):
    client_ip = request.client.host
    logging.info(f"Starting model training for IP: {client_ip}")
    if client_ip not in logged_in_users:
        logging.warning(f"Unauthorized access to training initiation from IP: {client_ip}")
        return RedirectResponse(url="/developer/train-model/login", status_code=303)

    try:
        logging.info("Initializing model training pipeline.")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logging.info("Model training started successfully.")
        return templates.TemplateResponse("developer_check.html", {"request": request, "message": "Model training started successfully!"})
    
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        return templates.TemplateResponse("developer_check.html", {"request": request, "message": f"Model training failed: {str(e)}"})
    
# Cost prediction route
@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        logging.info("Received request for cost prediction.")
        form = DataForm(request)
        await form.get_shipping_data()
        logging.info("Collected shipping data from form.")
        # Create an instance of ShippingData with the extracted form data
        shipping_data = ShippingData(
            line_item_quantity=form.line_item_quantity,
            line_item_value=form.line_item_value,
            pack_price=form.pack_price,
            unit_price=form.unit_price,
            weight=form.weight,
            country=form.country,
            shipment_mode=form.shipment_mode,
            scheduled_delivery_date=form.scheduled_delivery_date,
            delivered_to_client_date=form.delivered_to_client_date,
            delivery_recorded_date=form.delivery_recorded_date,
            first_line_designation=form.first_line_designation,
            fulfill_via=form.fulfill_via,
            vendor_inco_term=form.vendor_inco_term,
            product_group=form.product_group,
            sub_classification=form.sub_classification,
            vendor=form.vendor,
            molecule_test_type=form.molecule_test_type,
            brand=form.brand,
            dosage=form.dosage,
            dosage_form=form.dosage_form,
            manufacturing_site=form.manufacturing_site,
            freight_cost=form.freight_cost,
            line_item_insurance=form.line_item_insurance,
            unit_of_measure=form.unit_of_measure
        )
        
        logging.info("Converted form data to ShippingData object.")

        cost_df = shipping_data.get_input_data_frame()
        logging.info("Generated input dataframe for cost prediction.")

        cost_predictor = CostPredictor()
        cost_value = round(cost_predictor.predict(X=cost_df)[0], 2)
        logging.info(f"Predicted cost: {cost_value}")
        
        return templates.TemplateResponse("price_prediction.html", {"request": request, "context": f"Predicted Cost: {cost_value}"})
    
    except Exception as e:
        logging.error(f"Error during cost prediction: {str(e)}")
        raise CustomException(str(e), sys)


# -------------------- RUN SERVER -------------------- #
# Run FastAPI app
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.info("Launching FastAPI app...")
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)
