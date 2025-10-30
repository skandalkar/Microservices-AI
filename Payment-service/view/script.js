/** Razorpay Integration Demonstration */

const Base_URL = "http://localhost:5003/api";

document.addEventListener('DOMContentLoaded', () => {
    // All 'Pay Now' buttons
    const payButtons = document.querySelectorAll('.pay-button');
    payButtons.forEach(button => {
        button.addEventListener('click', handlePaymentInitiation);
    });
});

/** Step 1: Initiates the payment process by calling the backend to create an order. */
async function handlePaymentInitiation(event) {

    const button = event.target;
    const card = button.closest('.product-card');

    if (!card) return;

    // 1. Get Product Data from HTML data attributes
    const price = card.getAttribute('data-price');
    const productName = card.getAttribute('data-name');
    const productId = card.getAttribute('data-id');

    button.textContent = 'Creating Order...';
    button.disabled = true;
    button.style.backgroundColor = '#FFC107'; // Yellow/Processing color

    try {
        //API
        const res = await fetch(`${Base_URL}/create-order`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                productId: productId,
                amount: price //Send to backend API
            })
        });

        const data = await res.json();

        if (data.success) {
            console.log("Order created successfully. Response: ", data);
            openRazorpayMoal(data, productName, button);
        } else {
            console.log("Order creation failed: ", data);
            alert(`Error while creating order: ${data.message}`);
            resetButton(button);
        }
    } catch (error) {
        console.log("Something went wrong, ", error)
        alert('A network error occurred.');
        resetButton(button);
    }
};

/** Step 2: Open Razorpay payment modal dialog */
function openRazorpayMoal(orderData, productName, button) {
    const options = {
        key: "My_Test_Key",
        amount: Math.round(orderData.amount * 100), 
        currency: orderData.currency,
        name: 'Payment Gateway Demo',
        description: `Payment for ${productName}`,
        order_id: orderData.orderId,
        handler: async function (response) {
            // Handler is called by Razorpay on successful payment
            button.textContent = 'Verifying Payment...';
            button.style.backgroundColor = '#2196F3'; 
           
            await verifyPayment(response, button);
        },
        modal: {
            ondismiss: function () {
                console.log("Razorpay modal closed by user. Payment cancelled.");
                resetButton(button);
            }
        },
        theme: {
            color: '#4CAF50'
        }
    };

    button.textContent = 'Awaiting Payment...';
    const rzp1 = new Razorpay(options);
    rzp1.open();
};

/** Step 3: Call API verify-payment to verify the payment signature */
async function verifyPayment(response, button) {

    try {
        //API 
        const verifyResponse = await fetch(`${Base_URL}/verify-payment`, {
            method: 'POST',
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                order_id: response.razorpay_order_id,
                payment_id: response.razorpay_payment_id,
                signature: response.razorpay_signature // Razorpay signature for verification
            })
        });

        const verifyData = await verifyResponse.json();

        if (verifyData.success) {
            alert('Paid Successfully and Verified!');
            console.log('Payment verified successfully. Response: ', verifyData);
            button.textContent = "Payment Successful!";
            button.style.backgroundColor = 'green';
        } else {
            alert('Verification failed, payment signature could not match for verification.');
            console.error("Verification failed: Response: ", verifyData);
            button.textContent = "Verification Failed";
            button.style.backgroundColor = 'red';
        }
    } catch (error) {
        console.error('Something went wrong, ', error);
        alert('Verification failed. check server logs.');
        button.textContent = "Payment Error";
        button.style.backgroundColor = 'red';
    }
};

/* reset the button state */
function resetButton(button) {
    button.textContent = 'Pay Now';
    button.disabled = false;
    button.style.backgroundColor = '#4CAF50';
}