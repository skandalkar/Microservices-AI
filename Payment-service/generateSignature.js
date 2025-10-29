/** This is Demo file to generate offline or test mode signature for payment verification */

const dotenv = require('dotenv');
dotenv.config()
const crypto = require("crypto");

    order_id = "order_RZNJwlUJFhnXWj";
    payment_id = "pay_demo12345";

    const body = order_id + "|" + payment_id;

    //object of hmac
    const generateSignature = crypto
        .createHmac("sha256", process.env.RAZORPAY_KEY_SECRET)
        .update(body.toString())
        .digest("hex");

    console.log(`Generated Signature: ${generateSignature}`)

console.log("Done.....\n")