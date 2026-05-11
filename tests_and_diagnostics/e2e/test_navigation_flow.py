# This automated E2E test uses Playwright to validate the navigation/usage flow of the
# entire app. More info here: https://sixfeetup.com/blog/end-to-end-testing-python-playwright.

# We test the following user workflow: A user opens the home page and can navigate from
# the home-page cards to Explore, Predict, and Documentation. On the Explore page, the
# snapshot cards populate and main interactive controls are visible. On the Predict page,
# selecting a state updates the county dropdown and the prediction form can be submitted.

from playwright.sync_api import expect

# Requirement: the Flask app must be running locally before the test is started.
BASE_URL = "http://127.0.0.1:5000"



def test_user_can_navigate_from_home_cards(page):
    page.goto(BASE_URL)

    assert page.get_by_text("Weather, health,").is_visible()
    assert page.get_by_text("Where would you like to go?").is_visible()

    page.get_by_role("link", name="EXPLORE DATA →").click()
    page.wait_for_url(f"{BASE_URL}/explore")

    assert page.get_by_text("Exploratory data analysis").is_visible()
    expect(page.locator("#snap-states")).not_to_have_text("—")
    expect(page.locator("#snap-counties")).not_to_have_text("—")
    expect(page.locator("#var-health")).to_be_visible()
    expect(page.locator("#var-weather")).to_be_visible()
    expect(page.locator("#yr-start")).to_be_visible()
    expect(page.locator("#yr-end")).to_be_visible()

    page.goto(BASE_URL)
    page.get_by_role("link", name="OPEN PREDICTOR →").click()
    page.wait_for_url(f"{BASE_URL}/predict")

    assert page.get_by_text("Health Outcome Predictor").is_visible()

    state_select = page.locator("#stateFilter")
    county_select = page.locator("#countySelect")

    state_select.select_option("CA")

    assert county_select.locator("option").count() > 1
    assert county_select.locator('option[data-state="CA"]').count() > 0

    county_select.select_option("ALAMEDA COUNTY|CA")
    page.locator('select[name="target"]').select_option("CASTHMA")
    page.locator('select[name="scenario"]').select_option("middle_road")

    page.get_by_role("button", name="Run prediction →").click()

    expect(page.get_by_text("CASTHMA — ALAMEDA COUNTY, CA")).to_be_visible()
    expect(page.get_by_text("Predicted Value (%)")).to_be_visible()
    expect(page.get_by_text("Lower 90% CI")).to_be_visible()
    expect(page.get_by_text("Upper 90% CI")).to_be_visible()
    expect(page.get_by_text("There was a prediction error")).not_to_be_visible()

    page.goto(BASE_URL)
    page.get_by_role("link", name="ABOUT THE DATA →").click()
    page.wait_for_url(f"{BASE_URL}/docs")

    assert page.get_by_role("heading", name="API Reference").is_visible()


# Note: Manual checks were also implemented, i.e., non-automated navigation and clicks
# on the website and on the different features and options.