const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const router = require('./routes/route.payments');
const app = express();

app.use(express.json())
app.use(cors());

dotenv.config();
const port = process.env.PORT || 5003;

//payment routes
app.use("/api",router)

app.get("/api", (req, res) =>{
    res.send("This is Payment Gateway Microservice.")
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});