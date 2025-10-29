const Razorpay = require("razorpay");
const dotenv = require("dotenv");
dotenv.config();

const razorInstance = new Razorpay({
  key_id: process.env.RAZORPAY_KEY_ID,
  key_secret: process.env.RAZORPAY_KEY_SECRET
});

module.exports = razorInstance;