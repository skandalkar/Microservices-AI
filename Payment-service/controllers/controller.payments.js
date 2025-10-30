const dotenv = require('dotenv');
dotenv.config()
const crypto = require("crypto");

const createRazorpayInstance = require("../config/razorpay.config");

const razorInstance = createRazorpayInstance;

const createOrder = async (req, res) => {
    try {
        // For creating order metadata required
        // This is only from frontend at production we need to fetch from database an actual price
        const { productId, amount } = req.body;

        const amountToRazor = Math.round(amount * 100); // always integer
        const options = {
            amount: amountToRazor,
            currency: "INR",
            receipt: `receipt_order_${productId}`
        };

        console.log("Creating Razorpay order with:", options);

        
        const order = await razorInstance.orders.create(options);

        //Order successful
        console.log("Order created successfully:", order);

        return res.status(200).json({
            success: true,
            orderId: order.id,
            amount: amountToRazor,
            currency: order.currency
        });

    } catch (error) {
        console.error("Razorpay Error:", error);
        return res.status(500).json({
            success: false,
            message: error?.message || "Something went wrong!"
        });
    }
};

//This is for verification of the payment status
const verifyPayment = async (req, res) => {
    const { order_id, payment_id, signature } = req.body;

    const body = order_id + "|" + payment_id;

    //object of hmac
    const generateSignature = crypto
        .createHmac("sha256", process.env.RAZORPAY_KEY_SECRET)
        .update(body.toString())
        .digest("hex");

    //compare signature and generateSignature
    if (generateSignature === signature) {
        return res.status(200).json({
            success: true,
            message: "Payment verified.",
        });
    }
    else {
        return res.status(400).json({
            success: false,
            message: "Payment not verified.",
        });
    }
}

module.exports = { createOrder, verifyPayment };