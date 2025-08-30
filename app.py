# Import FastAPI and related modules for creating API endpoints and handling HTTP requests
from fastapi import FastAPI,Request, HTTPException, Form, Depends 
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sys
import numpy as np
import uvicorn
import os 


# Import custom modules for model prediction, training, and logging
from src.Shipment_Price_Prediction.utils.main_utils import MainUtils

from src.Shipment_Price_Prediction.pipelines.prediction_pipeline import CostPredictor, ShippingData
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
        self.UTILS = MainUtils()

        # Numerical Variables
        self.line_item_quantity: Optional[int] = None
        self.pack_price: Optional[float] = None
        self.unit_price: Optional[float] = None
        self.weight: Optional[float] = None

        # Categorical Variables
        self.country: Optional[str] = None
        self.vendor: Optional[str] = None
        self.molecule_test_type: Optional[str] = None
        self.brand: Optional[str] = None
        self.dosage: Optional[str] = None
        self.dosage_form: Optional[str] = None
        self.manufacturing_site: Optional[str] = None
        self.managed_by: Optional[str] = None  
        self.fulfill_via: Optional[str] = None
        self.vendor_inco_term: Optional[str] = None
        self.shipment_mode: Optional[str] = None
        self.unit_of_measure: Optional[str] = None
        self.product_group: Optional[str] = None
        self.sub_classification: Optional[str] = None

        # Datetime Variables
        self.scheduled_delivery_date: Optional[str] = None
        self.delivered_to_client_date: Optional[str] = None
        self.delivery_recorded_date: Optional[str] = None

        # Binary Variables
        self.first_line_designation: Optional[int] = None

    async def get_shipping_data(self):
        logging.info("Extracting shipping data from form...")
        form = await self.request.form()

        # Numerical Variables
        self.line_item_quantity = self.UTILS.safe_int(form.get("line_item_quantity", "0"))
        self.pack_price = self.UTILS.safe_float(form.get("pack_price", "0.0"))
        self.unit_price = self.UTILS.safe_float(form.get("unit_price", "0.0"))
        self.weight = self.UTILS.safe_float(form.get("weight", "0.0"))

        # Categorical Variables
        self.country = form.get("country", "").strip()
        self.vendor = form.get("vendor", "").strip()
        self.molecule_test_type = form.get("molecule_test_type", "").strip()
        self.brand = form.get("brand", "").strip()
        self.dosage = form.get("dosage", "").strip()
        self.dosage_form = form.get("dosage_form", "").strip()
        self.manufacturing_site = form.get("manufacturing_site", "").strip()
        self.managed_by = form.get("managed_by", "").strip()
        self.fulfill_via = form.get("fulfill_via", "").strip()
        self.vendor_inco_term = form.get("vendor_inco_term", "").strip()
        self.shipment_mode = form.get("shipment_mode", "").strip()
        self.unit_of_measure = form.get("unit_of_measure", "").strip()
        self.product_group = form.get("product_group", "").strip()
        self.sub_classification = form.get("sub_classification", "").strip()

        # Datetime Variables
        self.scheduled_delivery_date = form.get("scheduled_delivery_date", "").strip()
        self.delivered_to_client_date = form.get("delivered_to_client_date", "").strip()
        self.delivery_recorded_date = form.get("delivery_recorded_date", "").strip()

        # Binary Variables
        self.first_line_designation = self.UTILS.safe_int(form.get("first_line_designation", "0"))

        
        
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
            # Numerical Variables
            line_item_quantity=form.line_item_quantity,
            pack_price=form.pack_price,
            unit_price=form.unit_price,
            weight=form.weight,

            # Categorical Variables
            country=form.country,
            vendor=form.vendor,
            molecule_test_type=form.molecule_test_type,
            brand=form.brand,
            dosage=form.dosage,
            dosage_form=form.dosage_form,
            manufacturing_site=form.manufacturing_site,
            managed_by=form.managed_by,
            fulfill_via=form.fulfill_via,
            vendor_inco_term=form.vendor_inco_term,
            shipment_mode=form.shipment_mode,
            unit_of_measure=form.unit_of_measure,
            product_group=form.product_group,
            sub_classification=form.sub_classification,

            # Datetime Variables
            scheduled_delivery_date=form.scheduled_delivery_date,
            delivered_to_client_date=form.delivered_to_client_date,
            delivery_recorded_date=form.delivery_recorded_date,

            # Binary Variables
            first_line_designation=form.first_line_designation
        )
        
        logging.info("Converted form data to ShippingData object.")

        cost_df = shipping_data.get_input_data_frame()
        logging.info("Generated input dataframe for cost prediction.")

        cost_predictor = CostPredictor()
        
        predicted_log_cost = cost_predictor.predict(X=cost_df)

        if isinstance(predicted_log_cost, (list, np.ndarray)):
            predicted_log_cost = predicted_log_cost[0]

        cost_value = round(float((predicted_log_cost)), 2)  

        logging.info(f"Predicted cost is: {cost_value}")

        # Render the prediction result page with the context
        return templates.TemplateResponse("prediction_result.html", {
            "request": request,

            # Numerical Variables
            "cost_value": cost_value,
            "line_item_quantity": form.line_item_quantity,
            "pack_price": form.pack_price,
            "unit_price": form.unit_price,
            "weight": form.weight,

            # Categorical Variables
            "country": form.country,
            "vendor": form.vendor,
            "molecule_test_type": form.molecule_test_type,
            "brand": form.brand,
            "dosage": form.dosage,
            "dosage_form": form.dosage_form,
            "manufacturing_site": form.manufacturing_site,
            "managed_by": form.managed_by,
            "fulfill_via": form.fulfill_via,
            "vendor_inco_term": form.vendor_inco_term,
            "shipment_mode": form.shipment_mode,
            "unit_of_measure": form.unit_of_measure,
            "product_group": form.product_group,
            "sub_classification": form.sub_classification,

            # Datetime Variables
            "scheduled_delivery_date": form.scheduled_delivery_date,
            "delivered_to_client_date": form.delivered_to_client_date,
            "delivery_recorded_date": form.delivery_recorded_date,

            # Binary Variables
            "first_line_designation": form.first_line_designation
        })

    
    except Exception as e:
        logging.error(f"Error during cost prediction: {str(e)}")
        raise CustomException(str(e), sys)


# -------------------- RUN SERVER -------------------- #
# Run FastAPI app
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.info("Launching FastAPI app...")
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=False)  # reload=True is fine for local and not while deployment.
