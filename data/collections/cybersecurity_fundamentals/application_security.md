# Application Security Fundamentals

## SQL Injection (SQLi)

SQL Injection occurs when untrusted user input is directly concatenated or interpolated into a dynamic database query without proper validation or parameterization. This allows an attacker to manipulate the structure of SQL commands executed by the database.

Consequences of SQL Injection:
- Bypassing authentication mechanisms (for example, supplying `' OR '1'='1` in login fields).
- Unauthorized extraction of sensitive records, including password hashes and personal data.
- Modifying, inserting, or deleting database records.
- In severe configurations, executing administrative operating system commands via database stored procedures.

Mitigation:
- Primary Defense: Parameterized queries (prepared statements), where SQL code is compiled prior to binding user parameters as literal data.
- Defense-in-depth: Object-Relational Mapping (ORM) frameworks with parameterized bindings, input validation, and database accounts operating with the principle of least privilege.

## Cross-Site Scripting (XSS)

Cross-Site Scripting occurs when an application includes untrusted data in an HTTP response sent to a web browser without adequate encoding or sanitization, causing the browser to execute malicious client-side JavaScript.

Types of XSS:
- Stored XSS (Persistent): The malicious script is permanently stored on the server (such as in a database comment or forum post) and delivered to victims when they view the page.
- Reflected XSS (Non-Persistent): The malicious script is injected into a request URL or parameter and immediately reflected back in the server response.
- DOM-based XSS: The vulnerability exists entirely in client-side script that processes data from an untrusted source (like `window.location`) in an unsafe manner without server involvement.

Mitigation:
- Contextual Output Encoding: Escaping characters such as `<`, `>`, `&`, `"`, and `'` before rendering user input in HTML body, attributes, or script contexts.
- Content Security Policy (CSP): An HTTP response header that restricts the sources from which scripts and other resources can be loaded and executed.
- Using modern frontend frameworks that automatically encode rendered variables.

## Cross-Site Request Forgery (CSRF)

Cross-Site Request Forgery is an attack that forces an authenticated user's browser to send forged HTTP requests to a vulnerable application that trusts the user's active session cookies.

Mitigation:
- Anti-CSRF Tokens: Generating unpredictable, unique, and secret tokens associated with the user session and verifying them on state-changing requests (POST, PUT, DELETE).
- SameSite Cookie Attribute: Configuring session cookies with `SameSite=Strict` or `SameSite=Lax` to prevent browsers from attaching session cookies on cross-origin requests.
- Re-authentication: Requiring password confirmation before performing critical account actions.
